import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env

from gym.envs.mujoco.jaco import JacoEnv


# Toss and serve
class JacoServeEnv(JacoEnv):
    def __init__(self, with_rot=0):
        super().__init__(with_rot=with_rot)

        # config
        self._config.update({
            "sparse_reward": 0,
            "guide_reward": 100,
            "pick_reward": 200,
            "release_reward": 50,
            "up_reward": 50,
            "pos_stable_reward": 2,
            "success_toss_reward": 100,
            "x_vel_reward": 1,
            "hit_reward": 100,
            "target_reward": 200,
            "return_reward": 500,
            "success_reward": 1000,
            "release_height": 0.7,
            "max_height": 2.0,
            "hit_height": 1.0,
            "return_height": 0.8,
            "target_height": 1.2,
            "hit_threshold": 0.1,
            "target_threshold": 0.2,
            "return_threshold": 0.1,
            "random_box": 0.04,
            "init_randomness": 0.005,
            "box_size": 0.04,
        })

        # state
        self._pick_height = 0
        self._boxtop = [0, 0, 0]
        self._dist_boxtop = 0
        self._picked = False
        self._released = False
        self._above = False
        self._falling = False
        self._success_toss = False
        self._max_height = 0
        self._hit_box = False
        self._hit_target = False
        self._min_dist_target = 10
        self._target_pos = [0, 0, 0]
        self._return_box_pos = np.array([0.4, 0.3, self._config["return_height"]])

        # env info
        self.reward_type += ["guide_reward", "pick_reward", "release_reward",
                             "up_reward", "pos_stable_reward",
                             "hit_reward", "x_vel_reward", "return_reward", "target_reward",
                             "success_reward", "success", "success_toss_reward"]
        self.ob_type = self.ob_shape.keys()

        mujoco_env.MujocoEnv.__init__(self, "jaco_serve.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        x_before = self._get_box_pos()[0]
        self.do_simulation(a, self.frame_skip)
        box_pos = self._get_box_pos()
        x_after = box_pos[0]

        ob = self._get_obs()
        done = False
        success = False

        guide_reward = 0
        pick_reward = 0
        pos_stable_reward = 0
        release_reward = 0
        up_reward = 0
        success_toss_reward = 0
        hit_reward = 0
        x_vel_reward = 0
        target_reward = 0
        return_reward = 0
        success_reward = 0
        ctrl_reward = self._ctrl_reward(a)

        hand_pos = self._get_hand_pos()
        box_z = box_pos[2]
        dist_hand = self._get_distance_hand('box')
        dist_boxtop = np.linalg.norm(hand_pos - self._boxtop)
        in_air = box_z > self._config["box_size"] + 0.03
        in_hand = dist_hand < 0.10
        on_ground = box_z < self._config["box_size"] + 0.03
        hit = dist_hand < 0.12
        x_vel = (x_after - x_before) / self.dt

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

        # pos stable during toss
        if self._picked and not self._hit_box:
            pos_diff = abs(box_pos[0] - 0.4) + abs(box_pos[1] - 0.3) - 0.5
            pos_stable_reward = -self._config["pos_stable_reward"] * pos_diff

        # falling
        if self._released and not self._falling:
            if self._max_height < box_z:
                self._max_height = box_z
            else:
                up_reward = self._config["up_reward"] * \
                    (1 - abs(self._config["max_height"] - box_z) / self._config["max_height"])
                if abs(box_z - self._config["max_height"]) < 0.1:
                    up_reward += self._config["up_reward"]
                    self._falling = True
                else:
                    done = True

        if self._falling and not self._hit_box and not self._success_toss and \
                (box_z < self._config["hit_height"] - self._config["hit_threshold"] or hit):
            self._success_toss = abs(box_pos[0] - 0.4) < 0.1 and abs(box_pos[1] - 0.3) < 0.1
            if self._success_toss:
                success_toss_reward = self._config["success_toss_reward"]

        if self._falling and hit and not self._hit_box:
            hit_reward = self._config["hit_reward"] * (1 - abs(box_z - self._config["hit_height"]))
            if abs(box_z - self._config["hit_height"]) < self._config["hit_threshold"]:
                self._hit_box = True
                hit_reward += self._config["hit_reward"]
                print('hit')
            else:
                done = True
                print('hit, but fail')

        if self._hit_box and not self._hit_target:
            x_vel_reward = -self._config["x_vel_reward"] * abs(x_vel)
            dist_target = self._get_distance('target', 'box')
            self._min_dist_target = min(self._min_dist_target, dist_target)

            if dist_target < self._config["target_threshold"]:
                print('hit target')
                self._hit_target = True
                target_reward = self._config["target_reward"]

                # For now, only care toss-serve
                success = True
                done = True
                success_reward = self._config["success_reward"]

        if self._hit_target:
            dist_hit_pos = np.linalg.norm(box_pos - self._return_box_pos)
            self._min_dist_hit_pos = min(self._min_dist_hit_pos, dist_hit_pos)

            if dist_hit_pos < self._config["return_threshold"]:
                print('return correctly')
                success = True
                success_reward = self._config["success_reward"]
                done = True
                return_reward = self._config["return_reward"]

        # fail
        if self._hit_box and on_ground:
            done = True
            if self._hit_box and not self._hit_target:
                target_reward = self._config["target_reward"] * \
                    (2 - self._min_dist_target) / 2
            if self._hit_target:
                return_reward = self._config["return_reward"] * \
                    (2 - self._min_dist_hit_pos) / 2

        if self._config["sparse_reward"] == 0:
            reward = ctrl_reward + guide_reward + pick_reward + release_reward + \
                up_reward + pos_stable_reward + hit_reward + x_vel_reward + \
                target_reward + return_reward + success_reward + success_toss_reward
        else:
            reward = 1 if self._success_toss or success else 0

        info = {"ctrl_reward": ctrl_reward,
                "pick_reward": pick_reward,
                "guide_reward": guide_reward,
                "release_reward": release_reward,
                "up_reward": up_reward,
                "success_toss_reward": success_toss_reward,
                "pos_stable_reward": pos_stable_reward,
                "hit_reward": hit_reward,
                "x_vel_reward": x_vel_reward,
                "target_reward": target_reward,
                "return_reward": return_reward,
                "success_reward": success_reward,
                "success": success}
        return ob, reward, done, info

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        qacc = self.data.qacc
        hand_pos = self._get_hand_pos()
        if self._with_rot == 0:
            qpos = qpos[:12]
            qvel = qvel[:12]
            qacc = qacc[:12]
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
        self.set_state(qpos, qvel)

    def reset_model(self):
        # reset jaco
        r = self._config["init_randomness"]
        qpos = self.init_qpos + np.random.uniform(low=-r, high=r, size=self.model.nq)
        qvel = self.init_qvel + np.random.uniform(low=-r, high=r, size=self.model.nv)
        self.set_state(qpos, qvel)

        # reset box
        self.reset_box()

        # set thresholds
        idx = self.model.body_name2id('release_point')
        self.model.body_pos[idx][2] = self._config["release_height"]
        idx = self.model.body_name2id('max_point')
        self.model.body_pos[idx][2] = self._config["max_height"]

        # set target
        idx = self.model.body_name2id('target')
        self.model.body_pos[idx][2] = self._config["target_height"]

        # init state
        self._picked = False
        self._released = False
        self._above = False
        self._falling = False
        self._success_toss = False
        self._hit_box = False
        self._hit_target = False

        self._dist_boxtop = np.linalg.norm(self._get_hand_pos() - self._boxtop)
        self._boxtop = self._init_box_pos + [0, 0, self._config["box_size"]]
        self._pick_height = 0
        self._max_height = 0
        self._min_dist_target = 5
        self._target_pos = self._get_pos('target')
        self._min_dist_hit_pos = 5

        return self._get_obs()

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

    def get_next_primitive(self, ob, prev_primitive):
        if self._released:
            return 'hit'
        else:
            return 'toss'
