import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from gym.envs.mujoco.jaco import JacoEnv


class JacoHitEnv(JacoEnv):
    def __init__(self, with_rot=0):
        super().__init__(with_rot=with_rot)

        # config
        self._config.update({
            "x_vel_reward": 1,
            "hit_reward": 100,
            "target_reward": 200,
            "return_reward": 500,
            "success_reward": 1000,
            "max_height": 2,
            "hit_height": 1.0,
            "return_height": 0.8,
            "target_height": 1.2,
            "hit_threshold": 0.1,
            "target_threshold": 0.2,
            "return_threshold": 0.1,
            "random_throw": 0.1,
            "random_box": 0.1,
            "init_randomness": 0.005,
            "box_size": 0.04,
            "random_steps": 10,
        })

        # state
        self._hit_box = False
        self._hit_target = False
        self._min_dist_target = 10
        self._target_pos = [0, 0, 0]
        self._return_box_pos = np.array([0.4, 0.3, self._config["return_height"]])

        # env info
        self.reward_type += ["hit_reward", "x_vel_reward", "return_reward",
                             "success", "success_reward", "target_reward"]
        self.ob_type = self.ob_shape.keys()

        mujoco_env.MujocoEnv.__init__(self, "jaco_hit.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        x_before = self._get_box_pos()[0]
        self.do_simulation(a, self.frame_skip)
        box_pos = self._get_box_pos()
        x_after = box_pos[0]

        ob = self._get_obs()
        done = False
        success = False
        hit_reward = 0
        x_vel_reward = 0
        return_reward = 0
        target_reward = 0
        success_reward = 0
        ctrl_reward = self._ctrl_reward(a)

        dist_hand = self._get_distance_hand('box')
        box_z = box_pos[2]
        on_ground = box_z < self._config["box_size"] + 0.03
        hit = dist_hand < 0.12
        x_vel = (x_after - x_before) / self.dt

        # hit a ball
        if hit and not self._hit_box:
            hit_reward = self._config["hit_reward"] * (1 - abs(box_z - self._config["hit_height"]))
            if abs(box_z - self._config["hit_height"]) < self._config["hit_threshold"]:
                self._hit_box = True
                hit_reward += self._config["hit_reward"]
                print('hit')
            else:
                done = True
                print('hit, but fail')

        # hit a target
        if self._hit_box and not self._hit_target:
            x_vel_reward = -self._config["x_vel_reward"] * abs(x_vel)
            dist_target = self._get_distance('target', 'box')
            self._min_dist_target = min(self._min_dist_target, dist_target)

            if dist_target < self._config["target_threshold"]:
                print('hit target')
                self._hit_target = True
                target_reward = self._config["target_reward"]
                done = success = True # temporary: finish after hitting traget
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
        if on_ground:
            done = True
            if self._hit_box and not self._hit_target:
                target_reward = self._config["target_reward"] * \
                    (2 - self._min_dist_target) / 2
            if self._hit_target:
                return_reward = self._config["return_reward"] * \
                    (2 - self._min_dist_hit_pos) / 2

        reward = ctrl_reward + hit_reward + x_vel_reward + \
            target_reward + return_reward + success_reward
        info = {"ctrl_reward": ctrl_reward,
                "hit_reward": hit_reward,
                "x_vel_reward": x_vel_reward,
                "target_reward": target_reward,
                "return_reward": return_reward,
                "success_reward": success_reward,
                "success": success}
        return ob, reward, done, info

    def _randomize_box(self):
        # set initial force
        force = np.random.uniform(-1, 1, size=3) * self._config["random_throw"]

        # apply force
        box_body_idx = self.model.body_name2id('box')
        xfrc = self.data.xfrc_applied
        xfrc[box_body_idx, :3] = force
        self.do_simulation(self.action_space.sample(), self.frame_skip)

        # reset force
        xfrc[box_body_idx] = 0

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

    def reset_model(self):
        r = self._config["init_randomness"]
        qpos = self.init_qpos + np.random.uniform(-r, r, size=self.model.nq)
        qvel = self.init_qvel + np.random.uniform(-r, r, size=self.model.nv)

        # set box's initial position
        r = self._config["random_box"]
        qpos[9:12] = np.array([0.4, 0.3, self._config["max_height"]]) + \
            np.random.uniform(-r, r, 3)

        # set box's initial veolicty
        r = self._config["random_throw"]
        qvel[9:12] += np.random.uniform(-r, r, 3)

        self.set_state(qpos, qvel)

        # set target
        idx = self.model.body_name2id('target')
        self.model.body_pos[idx][2] = self._config["target_height"]

        #self._randomize_box()
        self._hit_box = False
        self._hit_target = False
        self._min_dist_target = 5
        self._target_pos = self._get_pos('target')
        self._min_dist_hit_pos = 5
        self._return_box_pos = np.array([0.4, 0.3, self._config["return_height"]])

        # more perturb
        for _ in range(int(self._config["random_steps"])):
            self.do_simulation(self.unwrapped.action_space.sample(), self.frame_skip)

        return self._get_obs()

    def is_terminate(self, ob, init=False, env=None):
        t_pos = env.unwrapped._get_pos('target')
        b_pos = ob[9:12]
        if np.linalg.norm(t_pos - b_pos) < 0.1:
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
