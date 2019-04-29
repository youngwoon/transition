import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env

from gym.envs.mujoco.walker2d import Walker2dEnv


class Walker2dJumpEnv(Walker2dEnv):
    def __init__(self):
        super().__init__()

        # config
        self._config.update({
            "x_vel_reward": 2,
            "alive_reward": 1,
            "angle_reward": 0.1,
            "foot_reward": 0.01,
            "height_reward": 2,
            "jump_reward": 100,
            "pass_reward": 200,
            "success_reward": 100,
            "collision_penalty": 10,
            "x_vel_limit": 3,
            "y_vel_limit": 2,
            "success_dist_after_curb": 1.5,
            "random_steps": 5,
            "curb_height": 0.2,  # DO NOT CHANGE!
            "curb_randomness": 1.5,
            "apply_force": 100,
            "done_when_collide": 1,
        })

        # state
        self._curbs = None
        self._curbs_x = 7
        self._stage = 0
        self._pass_state = False
        self._post_curb_states = 0
        self._collide = False

        # env info
        self.reward_type += ["x_vel_reward", "alive_reward", "angle_reward",
                             "foot_reward", "height_reward",
                             "collision_penalty", "jump_reward", "pass_reward",
                             "success_reward", "success",
                             "x_vel_mean", "height_mean", "nz_mean", "delta_h_mean"]
        self.ob_shape.update({"curb": [2]})
        self.ob_type = self.ob_shape.keys()

        mujoco_env.MujocoEnv.__init__(self, "walker_v2.xml", 4)
        utils.EzPickle.__init__(self)

    def _get_curb_observation(self):
        if self._curbs is None:
            self._put_curbs()
        x_agent = self._get_walker2d_pos()
        #x_agent = self.data.qpos[0]
        self._stage = 0
        # where is the next curb
        pos = self._curbs['pos']
        size = self._curbs['size']
        if x_agent > pos + size + self._config["success_dist_after_curb"]:
            self._stage += 1
        curb_start = pos - size
        curb_end = pos + size
        if curb_start - x_agent > 5.0 or self._stage == 1:
            return (5.1, 5.2)
        return (curb_start - x_agent, curb_end - x_agent)

    def step(self, a):
        x_before = self.data.qpos[0]
        y_before = self.data.qpos[1]
        foot_before = min(self.data.body_xpos[4, 0], self.data.body_xpos[7, 0])
        right_foot_before = self.data.qpos[5]
        left_foot_before = self.data.qpos[8]
        self._get_curb_observation()
        stage_before = self._stage

        self.do_simulation(a, self.frame_skip)

        x_after = self.data.qpos[0]
        y_after = self.data.qpos[1]
        foot_after = min(self.data.body_xpos[4, 0], self.data.body_xpos[7, 0])
        right_foot_after = self.data.qpos[5]
        left_foot_after = self.data.qpos[8]
        self._get_curb_observation()
        stage_after = self._stage

        self._reset_external_force()
        if np.random.rand(1) < self._config["prob_apply_force"]:
            self._apply_external_force()

        pass_reward = 0
        success_reward = 0
        if stage_before < stage_after and not self._pass_state:
            pass_reward = self._config['pass_reward']
            self._pass_state = True

        collision_penalty = 0
        if self.collision_detection('curb'):
            self._collide = True
            collision_penalty = -self._config["collision_penalty"]

        x_vel_reward = 0
        angle_reward = 0
        height_reward = 0
        alive_reward = 0
        jump_reward = 0
        foot_reward = 0
        ctrl_reward = self._ctrl_reward(a)

        height = self.data.qpos[1]
        angle = self.data.qpos[2]
        delta_h = self.data.body_xpos[1, 2] - max(self.data.body_xpos[4, 2], self.data.body_xpos[7, 2])
        nz = np.cos(angle)
        x_vel = (x_after - x_before) / self.dt
        x_vel = self._config["x_vel_limit"] - abs(x_vel - self._config["x_vel_limit"])
        y_vel = (y_after - y_before) / self.dt
        y_vel = np.clip(y_vel, -self._config["y_vel_limit"], self._config["y_vel_limit"])
        right_foot_vel = abs(right_foot_after - right_foot_before) / self.dt
        left_foot_vel = abs(left_foot_after - left_foot_before) / self.dt

        # reward
        x_vel_reward = self._config["x_vel_reward"] * x_vel
        angle_reward = self._config["angle_reward"] * nz
        alive_reward = self._config["alive_reward"]
        height_reward = -self._config["height_reward"] * abs(1.1 - delta_h)
        foot_reward = -self._config["foot_reward"] * (right_foot_vel + left_foot_vel)

        # success
        success = False
        x_agent = self._get_walker2d_pos()

        c_pos = self._curbs['pos']
        c_size = self._curbs['size']
        for i, x in enumerate([c_pos - c_size, c_pos, c_pos + c_size]):
            if foot_before <= x and x < foot_after:
                pass_reward += self._config["pass_reward"] * (i + 1) / 3
            if x_before <= x and x < x_after:
                jump_reward += self._config["jump_reward"] * y_vel

        if x_agent > 15:
            success = True
            success_reward = self._config["success_reward"]

        reward = alive_reward + ctrl_reward + collision_penalty + pass_reward + \
            x_vel_reward + height_reward + angle_reward + success_reward + \
            jump_reward + foot_reward

        # fail or success
        done = height < self._config["min_height"] or success or (self._config["done_when_collide"] and self._collide)

        ob = self._get_obs()
        info = {"alive_reward": alive_reward,
                "ctrl_reward": ctrl_reward,
                "collision_penalty": collision_penalty,
                "x_vel_reward": x_vel_reward,
                "angle_reward": angle_reward,
                "height_reward": height_reward,
                "foot_reward": foot_reward,
                "jump_reward": jump_reward,
                "pass_reward": pass_reward,
                "success_reward": success_reward,
                "delta_h_mean": delta_h,
                "nz_mean": nz,
                "x_vel_mean": (x_after - x_before) / self.dt,
                "height_mean": height,
                "success": success}
        return ob, reward, done, info

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        qacc = self.data.qacc
        curb_obs = self._get_curb_observation()
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10), qacc, curb_obs]).ravel()

    def get_ob_dict(self, ob):
        if len(ob.shape) > 1:
            return {
                'joint': ob[:, :17],
                'acc': ob[:, 17:26],
                'curb': ob[:, -2:],
            }
        return {
            'joint': ob[:17],
            'acc': ob[17:26],
            'curb': ob[-2:],
        }

    def reset_model(self):
        self._post_curb_states = 0
        self._collide = False
        self._put_curbs()
        self._pass_state = False
        r = self._config["init_randomness"]
        self.set_state(
            self.init_qpos + np.random.uniform(low=-r, high=r, size=self.model.nq),
            self.init_qvel + np.random.uniform(low=-r, high=r, size=self.model.nv)
        )

        # more perturb
        for _ in range(int(self._config["random_steps"])):
            self.do_simulation(self.unwrapped.action_space.sample(), self.frame_skip)
        return self._get_obs()

    def _put_curbs(self):
        idx = self.model.geom_name2id('curb1')
        r = self._config["curb_randomness"]
        h = self._config["curb_height"]
        self.model.geom_pos[idx][0] = self._curbs_x + np.random.uniform(low=-r, high=r)
        self.model.geom_pos[idx][2] = h
        self.model.geom_size[idx][2] = h

        pos = self.model.geom_pos[idx][0]
        size = self.model.geom_size[idx][0]
        self._curbs = {'pos': pos, 'size': size}

    def is_terminate(self, ob, init=False, env=None):
        if init:
            self._passed = False
        if ob[26] < 0:
            self._passed = True
        if ob[26] >= 5.1 and self._passed:
            return True
        return False
