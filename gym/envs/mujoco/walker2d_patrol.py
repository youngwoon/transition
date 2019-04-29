import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env

from gym.envs.mujoco.walker2d import Walker2dEnv


# Forward and backward
class Walker2dPatrolEnv(Walker2dEnv):
    def __init__(self):
        super().__init__()

        # config
        self._config.update({
            "x_vel_reward": 2,
            "alive_reward": 1,
            "angle_reward": 0.1,
            "foot_reward": 0.01,
            "height_reward": 2,
            "success_reward": 100,
            "x_vel_limit": 3,
            "track_length": 2.0,
            "sparse_reward": 0,
            "with_balance": 1,
            "random_direction": 1,
            "forward_first": 1,
            "max_success": 5,
            "prob_apply_force": 0,
            "one_way_time_limit": 800,
        })

        # state
        self._direction = 1
        self._success_count = 0
        self._one_way_time_limit = self._config["one_way_time_limit"]

        self._balance_period = False

        # env info
        self.reward_type += ["x_vel_reward", "nz_mean", "delta_h_mean", "success",
                             "x_vel_mean", "height_mean",
                             "angle_reward", "height_reward", "alive_reward",
                             "foot_reward", "success_reward", "direction"]
        self.ob_shape.update({"distance": [1]})
        self.ob_type = self.ob_shape.keys()

        mujoco_env.MujocoEnv.__init__(self, 'walker_v1.xml', 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        x_before = self.data.qpos[0]
        right_foot_before = self.data.qpos[5]
        left_foot_before = self.data.qpos[8]

        self.do_simulation(a, self.frame_skip)

        x_after = self.data.qpos[0]
        right_foot_after = self.data.qpos[5]
        left_foot_after = self.data.qpos[8]

        self._reset_external_force()
        if np.random.rand(1) < self._config["prob_apply_force"]:
            self._apply_external_force()

        done = False
        success = False
        x_vel_reward = 0
        angle_reward = 0
        height_reward = 0
        alive_reward = 0
        foot_reward = 0
        success_reward = 0
        ctrl_reward = self._ctrl_reward(a)

        height = self.data.qpos[1]
        angle = self.data.qpos[2]
        delta_h = self.data.body_xpos[1, 2] - max(self.data.body_xpos[4, 2], self.data.body_xpos[7, 2])
        nz = np.cos(self.data.qpos[2])
        x_vel = (x_after - x_before) / self.dt * self._direction
        x_vel = self._config["x_vel_limit"] - abs(x_vel - self._config["x_vel_limit"])
        right_foot_vel = abs(right_foot_after - right_foot_before) / self.dt
        left_foot_vel = abs(left_foot_after - left_foot_before) / self.dt

        x_vel_reward = self._config["x_vel_reward"] * x_vel
        angle_reward = self._config["angle_reward"] * nz
        height_reward = -self._config["height_reward"] * abs(1.1 - delta_h)
        alive_reward = self._config["alive_reward"]
        foot_reward = -self._config["foot_reward"] * (right_foot_vel + left_foot_vel)

        # fail
        done = height < self._config["min_height"]
        self._one_way_time_limit -= 1
        if self._one_way_time_limit == 0:
            print('failed to patrol within a given time')
            done = True

        # success
        if self._direction * x_after > self._config["track_length"]:
            if x_vel < self._config["x_vel_limit"] - 1:
                done = True
            else:
                success = True
                self._success_count += 1
                self._direction *= -1
                self._balance_period = True
                self._one_way_time_limit = self._config["one_way_time_limit"]
                success_reward = self._config["success_reward"]
                print('success turn {} times'.format(self._success_count))

        if self._success_count == int(self._config["max_success"]):
            done = True
            print('Done (success {} times)'.format(self._success_count))

        if self._config["sparse_reward"] == 0:
            reward = x_vel_reward + angle_reward + height_reward + \
                ctrl_reward + alive_reward + foot_reward + success_reward
        else:
            reward = float(success)

        ob = self._get_obs()
        info = {"x_vel_reward": x_vel_reward,
                "ctrl_reward": ctrl_reward,
                "angle_reward": angle_reward,
                "height_reward": height_reward,
                "alive_reward": alive_reward,
                "foot_reward": foot_reward,
                "success_reward": success_reward,
                "delta_h_mean": delta_h,
                "nz_mean": nz,
                "x_vel_mean": abs((x_after - x_before) / self.dt),
                "height_mean": height,
                "success": success,
                "direction": self._direction}
        return ob, reward, done, info

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        qacc = self.data.qacc
        target = self._direction * self._config['track_length']
        distance = target - qpos[0]
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10), qacc, [distance]])

    def get_ob_dict(self, ob):
        if len(ob.shape) > 1:
            return {
                'joint': ob[:, :17],
                'acc': ob[:, 17:26],
                'distance': ob[:, -1:],
            }
        return {
            'joint': ob[:17],
            'acc': ob[17:26],
            'distance': ob[-1:],
        }

    def reset_model(self):
        init_randomness = self._config["init_randomness"]
        qpos = self.init_qpos + np.random.uniform(low=-init_randomness,
                                                  high=init_randomness,
                                                  size=self.model.nq)
        qvel = self.init_qvel + np.random.uniform(low=-init_randomness,
                                                  high=init_randomness,
                                                  size=self.model.nv)
        # reset state
        self._success_count = 0
        if self._config["random_direction"] > 0:
            if np.random.rand(1) > 0.5:
                self._direction = 1
            else:
                self._direction = -1
        else:
            if self._config["forward_first"] > 0:
                self._direction = 1
            else:
                self._direction = -1

        qpos[0] = -self._direction * self._config["track_length"]
        self.set_state(qpos, qvel)

        # init target
        self._set_pos('target_forward', (self._config['track_length'], 0, 0))
        self._set_pos('target_backward', (-self._config['track_length'], 0, 0))

        # more perturb
        for _ in range(int(self._config["random_steps"])):
            self.do_simulation(self.unwrapped.action_space.sample(), self.frame_skip)

        # balance period
        self._balance_period = False
        self._one_way_time_limit = self._config["one_way_time_limit"]
        return self._get_obs()

    def get_next_primitive(self, ob, prev_primitive):
        if self._config["with_balance"] == 0:
            return "forward" if self._direction == 1 else "backward"
        else:
            if self._balance_period:
                self._balance_period = False
                return "balance"
            else:
                return "forward" if self._direction == 1 else "backward"
