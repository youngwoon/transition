import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env

from gym.envs.mujoco.jaco import JacoEnv


# Catch-and-catch
class JacoKeepCatchEnv(JacoEnv):
    def __init__(self, with_rot=1):
        super().__init__(with_rot=with_rot)

        # config
        self._config.update({
            "sparse_reward": 0,
            "catch_reward": 100,
            "hold_reward": 4,
            "random_throw": 1,
            "wait": 50,
            "init_randomness": 0.005,
            "max_success": 5,
            "sub_use_term_len": 50,
        })

        # state
        self._t = 0
        self._hold_duration = 0
        self._ep_t = -1
        self._dist_box = 0
        self._target_pos = [0, 0.2, 0.2]
        self.qpos_box = [0, 2, 1.5, 1, 0, 0, 0]
        self.qvel_box = [0, 0, 0, 0, 0, 0]

        # env info
        self.reward_type += ["catch_reward", "hold_reward", "success"]
        self.ob_type = self.ob_shape.keys()

        mujoco_env.MujocoEnv.__init__(self, "jaco_pick.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        self._t += 1
        self._ep_t += 1
        if self._ep_t == 1 or (self._ep_t != self._t and self._t == self._config["wait"]):
            self._throw_box()
        elif self._ep_t != self._t and self._t < self._config["wait"]:
            self._set_box()

        self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        done = False
        success = False
        reset = False
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
            reset = True

        # catch
        if in_air and in_hand:
            self._hold_duration += 1
            catch_reward = self._config["catch_reward"]
            dist = np.linalg.norm(self._target_pos - self._get_box_pos())
            hold_reward = self._config["hold_reward"] * (1 - dist)

            # success
            if self._hold_duration == self._config['sub_use_term_len']:
                reset = True
                success = True
                self._success_count += 1
                if self._success_count == int(self._config["max_success"]):
                    done = True

        # unstable simulation
        if self._fail:
            done = True
            self._fail = False

        if done:
            print('success {} times'.format(self._success_count))

        if reset:
            self.reset_box()

        if self._config["sparse_reward"] == 0:
            reward = ctrl_reward + catch_reward + hold_reward
        else:
            reward = float(success)

        info = {"ctrl_reward": ctrl_reward,
                "catch_reward": catch_reward,
                "hold_reward": hold_reward,
                "success": success}
        return ob, reward, done, info

    def _set_box(self):
        # set box's initial position
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()
        qpos[9:16] = self.qpos_box
        qvel[9:15] = self.qvel_box
        self.set_state(qpos, qvel)

    def _throw_box(self):
        self._set_box()

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

    def reset_box(self):
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()

        # set box's initial pose
        qpos[9:12] = np.asarray([0, 2.0, 1.5])
        init_randomness = self._config["init_randomness"]
        qpos[12:16] = self.init_qpos[12:16] + np.random.uniform(low=-init_randomness,
                                                                high=init_randomness,
                                                                size=4)
        qvel[9:15] = self.init_qvel[9:15] + np.random.uniform(low=-init_randomness,
                                                              high=init_randomness,
                                                              size=6)

        self.qpos_box = qpos[9:16]
        self.qvel_box = qvel[9:15]
        self.set_state(qpos, qvel)

        self._t = 0
        self._hold_duration = 0

    def reset_model(self):
        init_randomness = self._config["init_randomness"]
        qpos = self.init_qpos + np.random.uniform(low=-init_randomness,
                                                  high=init_randomness,
                                                  size=self.model.nq)
        qvel = self.init_qvel + np.random.uniform(low=-init_randomness,
                                                  high=init_randomness,
                                                  size=self.model.nv)
        self.set_state(qpos, qvel)

        self.reset_box()

        self._ep_t = 0
        self._success_count = 0
        return self._get_obs()

    def get_next_primitive(self, ob, prev_primitive):
        return 'catch'

