import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env

from gym.envs.mujoco.jaco import JacoEnv


class JacoCatchEnv(JacoEnv):
    def __init__(self, with_rot=1):
        super().__init__(with_rot=with_rot)

        # config
        self._config.update({
            "catch_reward": 100,
            "hold_reward": 4,
            "hold_duration": 50,
            "random_throw": 1,
            "init_randomness": 0.01,
            "random_steps": 10,
        })

        # state
        self._hold_duration = 0
        self._target_pos = [0.5, 0.2, 0.2]

        # env info
        self.reward_type += ["catch_reward", "hold_reward", "success"]
        self.ob_type = self.ob_shape.keys()

        mujoco_env.MujocoEnv.__init__(self, "jaco_pick.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        done = False
        success = False
        catch_reward = 0
        hold_reward = 0
        ctrl_reward = self._ctrl_reward(a)

        dist_box = self._get_distance_hand('box')
        box_z = self._get_box_pos()[2]
        in_hand = dist_box < 0.06
        in_air = box_z > 0.05
        on_ground = box_z <= 0.05

        # fail
        if on_ground:
            done = True

        # catch
        if in_air and in_hand:
            self._hold_duration += 1
            catch_reward = self._config["catch_reward"]
            dist = np.linalg.norm(self._target_pos - self._get_box_pos())
            hold_reward = self._config["hold_reward"] * (1 - dist)

            # success
            if self._hold_duration == self._config['hold_duration']:
                print('success catch! {}'.format(self._get_box_pos()))
                done = success = True

        reward = ctrl_reward + catch_reward + hold_reward
        info = {"ctrl_reward": ctrl_reward,
                "catch_reward": catch_reward,
                "hold_reward": hold_reward,
                "success": success}
        return ob, reward, done, info

    def _throw_box(self):
        # set initial force
        box_pos = self._get_box_pos()
        jaco_pos = self._get_pos('jaco_link_base')
        dx = 0.4 + np.random.uniform(0, 0.1) * self._config["random_throw"]
        dy = 0.3 + np.random.uniform(0, 0.1) * self._config["random_throw"]
        force = jaco_pos + [dx, dy, 1] - box_pos
        force = 110 * (force / np.linalg.norm(force))

        # apply force
        box_body_idx = self.model.body_name2id('box')
        xfrc = self.data.xfrc_applied
        xfrc[box_body_idx, :3] = force
        self.do_simulation(self.unwrapped.action_space.sample(), self.frame_skip)

        # reset force
        xfrc[box_body_idx] = 0

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        qacc = self.data.qacc
        hand_pos = self._get_hand_pos()
        return np.concatenate([qpos, np.clip(qvel, -30, 30), qacc, hand_pos]).ravel()

    def get_ob_dict(self, ob):
        if len(ob.shape) > 1:
            return {
                'joint': ob[:, :31],
                'acc': ob[:, 31:46],
                'hand': ob[:, 46:49],
            }
        else:
            return {
                'joint': ob[:31],
                'acc': ob[31:46],
                'hand': ob[46:49],
            }

    def reset_model(self):
        init_randomness = self._config["init_randomness"]
        qpos = self.init_qpos + np.random.uniform(low=-init_randomness,
                                                  high=init_randomness,
                                                  size=self.model.nq)
        qvel = self.init_qvel + np.random.uniform(low=-init_randomness,
                                                  high=init_randomness,
                                                  size=self.model.nv)
        # set box's initial position
        qpos[9:12] = np.asarray([0, 2.0, 1.5])
        self.set_state(qpos, qvel)

        self._hold_duration = 0

        # more perturb
        for _ in range(int(self._config["random_steps"])):
            self.do_simulation(self.unwrapped.action_space.sample(), self.frame_skip)

        self._throw_box()
        return self._get_obs()

    def is_terminate(self, ob, success_length=50, init=False, env=None):
        if init:
            self.count_evaluate = 0
            self.success = True

        box_pos = ob[9:12]
        hand_pos = ob[46:49]
        dist_box = np.linalg.norm(box_pos - hand_pos)
        box_z = box_pos[2]
        in_hand = dist_box < 0.06
        in_air = box_z > 0.05
        on_ground = box_z <= 0.05

        if on_ground and self.count_evaluate > 0:
            self.success = False

        if in_air and in_hand:
            self.count_evaluate += 1

        return self.success and self.count_evaluate >= success_length

