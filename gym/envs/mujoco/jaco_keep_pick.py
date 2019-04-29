import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env

from gym.envs.mujoco.jaco import JacoEnv


# Pick-and-pick
class JacoKeepPickEnv(JacoEnv):
    def __init__(self, with_rot=1):
        super().__init__(with_rot=with_rot)

        # config
        self._config.update({
            "sparse_reward": 0,
            "pick_reward": 100,
            "hold_reward": 2,
            "guide_reward": 2,
            "success_reward": 1,
            "random_box": 1,
            "init_randomness": 0.005,
            "max_success": 5,
            "sub_use_term_len": 50,
        })

        # state
        self._t = 0
        self._hold_duration = 0
        self._picked = False
        self._pick_height = 0
        self._dist_box = 0

        # env info
        self.reward_type += ["guide_reward", "pick_reward", "hold_reward",
                             "success_reward", "success"]
        self.ob_type = self.ob_shape.keys()

        mujoco_env.MujocoEnv.__init__(self, "jaco_pick.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        self._t += 1
        self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        done = False
        success = False
        reset = False
        guide_reward = 0
        pick_reward = 0
        hold_reward = 0
        ctrl_reward = self._ctrl_reward(a)
        success_reward = 0

        hand_pos = self._get_hand_pos()
        box_z = self._get_box_pos()[2]
        dist_box = self._get_distance_hand('box')
        in_hand = dist_box < 0.06
        in_air = box_z > 0.05

        if in_hand and in_air:
            self._picked = True

            # pick up
            if self._pick_height < min(self._target_pos[2], box_z):
                pick_reward = self._config["pick_reward"] * \
                    (min(self._target_pos[2], box_z) - self._pick_height)
                self._pick_height = box_z

            # hold
            dist = np.linalg.norm(self._target_pos - self._get_box_pos())
            hold_reward = self._config["hold_reward"] * (1 - dist)
            self._hold_duration += 1

            # success
            if self._hold_duration >= self._config["sub_use_term_len"]:
                reset = True
                success = True
                success_reward = self._config["success_reward"] * (200 - self._t)
                self._success_count += 1
                if self._success_count == int(self._config["max_success"]):
                    done = True

        # guide hand to the box
        if not self._picked:
            guide_reward = self._config["guide_reward"] * (self._dist_box - dist_box)
            self._dist_box = dist_box

        if self._picked and not in_hand:
            done = True

        if self._t == 200:
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
            reward = ctrl_reward + pick_reward + hold_reward + guide_reward + success_reward
        else:
            reward = float(success)

        info = {"ctrl_reward": ctrl_reward,
                "pick_reward": pick_reward,
                "hold_reward": hold_reward,
                "guide_reward": guide_reward,
                "success_reward": success_reward,
                "success": success}
        return ob, reward, done, info

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
        self._init_box_pos = np.asarray(
            [0.5 + np.random.uniform(0, 0.1) * self._config["random_box"],
             0.2 + np.random.uniform(0, 0.1) * self._config["random_box"],
             0.03])
        qpos[9:12] = self._init_box_pos
        init_randomness = self._config["init_randomness"]
        qpos[12:16] = self.init_qpos[12:16] + np.random.uniform(low=-init_randomness,
                                                                high=init_randomness,
                                                                size=4)
        qvel[9:15] = self.init_qvel[9:15] + np.random.uniform(low=-init_randomness,
                                                              high=init_randomness,
                                                              size=6)
        self.set_state(qpos, qvel)

        self._t = 0
        self._hold_duration = 0
        self._pick_height = 0
        self._picked = False
        self._dist_box = np.linalg.norm(self._get_hand_pos() - self._init_box_pos)
        self._target_pos = self._init_box_pos.copy()
        self._target_pos[2] = 0.3

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

        self._success_count = 0
        return self._get_obs()

    def get_next_primitive(self, ob, prev_primitive):
        return 'pick'

