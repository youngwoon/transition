import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from gym.envs.mujoco.jaco import JacoEnv


class JacoTossEnv(JacoEnv):
    def __init__(self, with_rot=0):
        super().__init__(with_rot=with_rot)

        # config
        self._config.update({
            "guide_reward": 100,
            "pick_reward": 200,
            "release_reward": 50,
            "up_reward": 50,
            "pos_stable_reward": 2,
            "success_reward": 100,
            "release_height": 0.7,
            "max_height": 2.0,
            "random_box": 0.04,
            "init_randomness": 0.005,
            "box_size": 0.04,
            "random_steps": 0,
        })

        # state
        self._pick_height = 0
        self._boxtop = [0, 0, 0]
        self._dist_boxtop = 0
        self._picked = False
        self._released = False
        self._above = False
        self._falling = False
        self._max_height = 0

        # env info
        self.reward_type += ["guide_reward", "pick_reward", "release_reward",
                             "up_reward", "pos_stable_reward",
                             "success_reward", "success"]
        self.ob_type = self.ob_shape.keys()

        mujoco_env.MujocoEnv.__init__(self, "jaco_toss.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        done = False
        success = False

        guide_reward = 0
        pick_reward = 0
        pos_stable_reward = 0
        release_reward = 0
        up_reward = 0
        success_reward = 0
        ctrl_reward = self._ctrl_reward(a)

        hand_pos = self._get_hand_pos()
        box_pos = self._get_box_pos()
        box_z = box_pos[2]
        dist_hand = self._get_distance_hand('box')
        dist_boxtop = np.linalg.norm(hand_pos - self._boxtop)
        in_air = box_z > self._config["box_size"] + 0.03
        in_hand = dist_hand < 0.10

        # guide hand to top of box
        if not self._picked and not (in_hand and in_air):
            guide_reward = self._config["guide_reward"] * (self._dist_boxtop - dist_boxtop)
            self._dist_boxtop = dist_boxtop

        # pick
        if in_hand and not self._released and \
                self._pick_height < min(box_z, self._config["release_height"]):
            pick_reward = self._config["pick_reward"] * \
                (min(box_z, self._config["release_height"]) - self._pick_height)
            self._picked = True
            self._pick_height = box_z

        # fail
        if self._picked and not in_hand and not in_air:
            done = True

        # release
        if not self._released and box_z > self._config["release_height"]:
            if in_hand:
                done = True
            else:
                release_reward = self._config["release_reward"]
                self._released = True

        # pos stable and up reward
        if self._picked:
            pos_diff = abs(box_pos[0] - 0.4) + abs(box_pos[1] - 0.3) - 0.5
            pos_stable_reward = -self._config["pos_stable_reward"] * pos_diff

        if self._released:
            if self._max_height < box_z:
                self._max_height = box_z
            elif not self._falling:
                up_reward = self._config["up_reward"] * \
                    (1 - abs(self._config["max_height"] - box_z) / self._config["max_height"])
                if abs(box_z - self._config["max_height"]) < 0.1:
                    up_reward += self._config["up_reward"]
                    self._falling = True
                else:
                    done = True

        if self._falling and box_z < self._config["release_height"]:
            done = True
            success = abs(box_pos[0] - 0.4) < 0.1 and abs(box_pos[1] - 0.3) < 0.1
            if success:
                print('success')
                success_reward = self._config["success_reward"]

        reward = ctrl_reward + guide_reward + pick_reward + release_reward + \
            up_reward + pos_stable_reward + success_reward
        info = {"ctrl_reward": ctrl_reward,
                "pick_reward": pick_reward,
                "guide_reward": guide_reward,
                "release_reward": release_reward,
                "up_reward": up_reward,
                "pos_stable_reward": pos_stable_reward,
                "success_reward": success_reward,
                "success": success}
        return ob, reward, done, info

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        qacc = self.data.qacc
        if self._with_rot == 0:
            qpos = qpos[:12]
            qvel = qvel[:12]
            qacc = qacc[:12]
        hand_pos = self._get_hand_pos()
        return np.concatenate([qpos, np.clip(qvel, -30, 30), qacc, hand_pos]).ravel()

    def get_ob_dict(self, ob):
        if len(ob.shape) > 1:
            if self._with_rot == 0:
                return {
                    'joint': ob[:, :24],
                    'acc': ob[:, 24:36],
                    'hand': ob[:, 36:39],
                }
            return {
                'joint': ob[:, :31],
                'acc': ob[:, 31:46],
                'hand': ob[:, 46:49],
            }
        else:
            if self._with_rot == 0:
                return {
                    'joint': ob[:24],
                    'acc': ob[24:36],
                    'hand': ob[36:39],
                }
            return {
                'joint': ob[:31],
                'acc': ob[31:46],
                'hand': ob[46:49]
            }

    def reset_box(self):
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()

        # set box's initial position
        r = self._config["random_box"]
        self._init_box_pos = np.asarray([0.4 + np.random.uniform(-r, r),
                                         0.3 + np.random.uniform(-r, r),
                                         self._config["box_size"]])
        qpos[9:12] = self._init_box_pos

        qvel[9:12] += np.random.uniform(low=-.005, high=.005, size=3)
        self.set_state(qpos, qvel)

    def reset_model(self):
        r = self._config["init_randomness"]
        qpos = self.init_qpos + np.random.uniform(-r, r, size=self.model.nq)
        qvel = self.init_qvel + np.random.uniform(-r, r, size=self.model.nv)
        self.set_state(qpos, qvel)

        # set thresholds
        idx = self.model.body_name2id('release_point')
        self.model.body_pos[idx][2] = self._config["release_height"]
        idx = self.model.body_name2id('max_point')
        self.model.body_pos[idx][2] = self._config["max_height"]

        self.reset_box()

        # more perturb
        for _ in range(int(self._config["random_steps"])):
            self.do_simulation(self.unwrapped.action_space.sample(), self.frame_skip)

        self._pick_height = 0
        self._boxtop = self._init_box_pos + [0, 0, self._config["box_size"]]
        self._dist_boxtop = np.linalg.norm(self._get_hand_pos() - self._boxtop)
        self._picked = False
        self._released = False
        self._above = False
        self._falling = False
        self._max_height = 0

        return self._get_obs()

    def is_terminate(self, ob, init=False, env=None):
        box_pos = ob[9:12]
        if box_pos[2] > env.unwrapped._config["release_height"]:
            return True
        return False

    def viewer_setup(self):
        # self.viewer.cam.trackbodyid = 1
        self.viewer.cam.trackbodyid = -1
        # self.viewer.cam.distance = self.model.stat.extent * 2
        self.viewer.cam.distance = 4
        self.viewer.cam.azimuth = 100
        # self.viewer.cam.azimuth = 90
        self.viewer.cam.lookat[0] = 0.5
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.elevation = -20
