import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from gym.envs.mujoco.walker2d import Walker2dEnv


class Walker2dBalanceEnv(Walker2dEnv):
    def __init__(self):
        super().__init__()

        # config
        self._config.update({
            "x_vel_reward": 1,
            "alive_reward": 1,
            "angle_reward": 0.1,
            "foot_reward": 0,
            "height_reward": 0.5,
            "random_steps": 10,
            "apply_force": 400,
            "prob_apply_force": 0.1,
        })

        # state
        self.t = 0

        # env info
        self.reward_type += ["x_vel_reward", "alive_reward", "angle_reward",
                             "foot_reward", "height_reward", "success",
                             "x_vel_mean", "height_mean", "delta_h_mean", "nz_mean"]
        self.ob_type = self.ob_shape.keys()

        mujoco_env.MujocoEnv.__init__(self, "walker_v1.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        x_before = self.data.qpos[0]
        right_foot_before = self.data.qpos[5]
        left_foot_before = self.data.qpos[8]
        self.do_simulation(a, self.frame_skip)
        x_after = self.data.qpos[0]
        right_foot_after = self.data.qpos[5]
        left_foot_after = self.data.qpos[8]

        done = False
        x_vel_reward = 0
        alive_reward = 0
        angle_reward = 0
        height_reward = 0
        foot_reward = 0
        ctrl_reward = self._ctrl_reward(a)

        # reward
        height = self.data.qpos[1]
        angle = self.data.qpos[2]
        delta_h = self.data.body_xpos[1, 2] - max(self.data.body_xpos[4, 2], self.data.body_xpos[7, 2])
        nz = np.cos(angle)
        x_vel = (x_after - x_before) / self.dt
        right_foot_vel = abs(right_foot_after - right_foot_before) / self.dt
        left_foot_vel = abs(left_foot_after - left_foot_before) / self.dt

        x_vel_reward = -self._config["x_vel_reward"] * np.abs(x_vel)
        alive_reward = self._config["alive_reward"]
        angle_reward = self._config["angle_reward"] * nz
        height_reward = -self._config["height_reward"] * abs(1.1 - delta_h)
        foot_reward = -self._config["foot_reward"] * (right_foot_vel + left_foot_vel)
        reward = x_vel_reward + angle_reward + height_reward + \
            ctrl_reward + alive_reward + foot_reward

        # fail
        done = height < self._config["min_height"]
        self.t += 1
        success = not done and self.t >= 1000
        if success: done = True

        self._reset_external_force()
        if np.random.rand(1) < self._config["prob_apply_force"]:
            self._apply_external_force()
            if self.t > 1: # to prevent mujoco initialization error
                self.do_simulation(self.unwrapped.action_space.sample(), self.frame_skip)

        ob = self._get_obs()
        info = {"alive_reward": alive_reward,
                "x_vel_reward": x_vel_reward,
                "ctrl_reward": ctrl_reward,
                "angle_reward": angle_reward,
                "height_reward": height_reward,
                "foot_reward": foot_reward,
                "delta_h_mean": delta_h,
                "nz_mean": nz,
                "height_mean": height,
                "success": success}
        return ob, reward, done, info

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        qacc = self.data.qacc
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10), qacc]).ravel()

    def get_ob_dict(self, ob):
        if len(ob.shape) > 1:
            return {'joint': ob[:, :17], 'acc': ob[:, 17:26]}
        else:
            return {'joint': ob[:17], 'acc': ob[17:26]}

    def reset_model(self):

        init_randomness = self._config["init_randomness"]
        qpos = self.init_qpos + np.random.uniform(low=-init_randomness,
                                                  high=init_randomness,
                                                  size=self.model.nq)
        qvel = self.init_qvel + np.random.uniform(low=-init_randomness,
                                                  high=init_randomness,
                                                  size=self.model.nv)
        self.set_state(qpos, qvel)

        # init target
        self._set_pos('target_forward', (0, 0, 0))
        self._set_pos('target_backward', (0, 0, 0))

        # more perturb
        for _ in range(int(self._config["random_steps"])):
            self.do_simulation(self.unwrapped.action_space.sample(), self.frame_skip)
        self.t = 0

        return self._get_obs()

    def is_terminate(self, ob, success_length=100, init=False, env=None):
        if init:
            self.count_evaluate = 0
            self.success = True
        height = ob[0]
        angle = ob[1]
        success = height > 0.8

        if success:
            self.count_evaluate += 1
        else:
            self.success = False

        return self.success and self.count_evaluate >= success_length
