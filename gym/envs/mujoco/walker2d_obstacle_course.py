import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env

from gym.envs.mujoco.walker2d import Walker2dEnv


# Forward, Lower, and jump
class Walker2dObstacleCourseEnv(Walker2dEnv):
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
            "collision_penalty": 10,
            "x_vel_limit": 3,
            "y_vel_limit": 2,
            "curb_height": 0.2, # DO NOT CHANGE!
            "ceil_height": 1.0,
            "ceil_size": 1.5, # DO NOT CHANGE!
            "obs_randomness": 0.5,
            "done_when_collide": 1,
            "sparse_reward": 0,
            "ob_detect_dist": 3.0,
            "success_dist_after_curb": 1.5,
            "prob_apply_force": 0,
        })

        # state
        self._obstacle_type = None
        self._obstacle_pos = None
        self._obstacle_size = None
        self._stage = 0
        self._curb_stage = 0
        self._num_curbs = 3
        self._ceil_stage = 0
        self._num_ceils = 2
        self._pass_state = [False] * (self._num_curbs + self._num_ceils)
        self._possible_x = np.array([8, 19, 30, 41, 52])
        self._success_count = 0
        self._meta_policy_stage = -1
        self._interval_time = 0
        self._interval_start_pos = 0

        # env info
        self.reward_type += ["x_vel_reward", "alive_reward", "angle_reward",
                             "foot_reward", "height_reward", "collision_penalty",
                             "jump_reward", "pass_reward", "success",
                             "x_vel_mean", "height_mean", "nz_mean", "delta_h_mean"]
        self.ob_shape.update({"curb": [2]})
        self.ob_type = self.ob_shape.keys()

        mujoco_env.MujocoEnv.__init__(self, 'walker_v2.xml', 4)
        utils.EzPickle.__init__(self)

    def _get_obstacles_observation(self):
        if self._obstacle_type is None:
            self._put_obstacles()
        x_agent = self._get_walker2d_pos()

        pos = self._obstacle_pos[self._stage]
        size = self._obstacle_size[self._stage]
        if self._obstacle_type[self._stage]:
            if x_agent > pos + size + self._config["success_dist_after_curb"]:
                self._stage += 1
                self._curb_stage += 1
        else:
            if x_agent > pos + size + 0.15:
                self._stage += 1
                self._ceil_stage += 1

        if self._obstacle_type[self._stage]:
            obs_obs = (5.1, 5.2)
        else:
            obs_obs = (5.1, 6.0)
        obs_start = self._obstacle_pos[self._stage] - \
                        self._obstacle_size[self._stage]
        obs_end = self._obstacle_pos[self._stage] + \
                        self._obstacle_size[self._stage]
        if obs_start - x_agent < self._config['ob_detect_dist']:
            obs_obs = (obs_start - x_agent, obs_end - x_agent)
        return obs_obs

    def step(self, a):
        self._interval_time += 1
        x_before = self.data.qpos[0]
        y_before = self.data.qpos[1]
        foot_before = min(self.data.body_xpos[4, 0], self.data.body_xpos[7, 0])
        right_foot_before = self.data.qpos[5]
        left_foot_before = self.data.qpos[8]
        self._get_obstacles_observation()
        stage_before = self._stage

        self.do_simulation(a, self.frame_skip)

        x_after = self.data.qpos[0]
        y_after = self.data.qpos[1]
        foot_after = min(self.data.body_xpos[4, 0], self.data.body_xpos[7, 0])
        right_foot_after = self.data.qpos[5]
        left_foot_after = self.data.qpos[8]
        self._get_obstacles_observation()
        stage_after = self._stage

        self._reset_external_force()
        if np.random.rand(1) < self._config["prob_apply_force"]:
            self._apply_external_force()

        done = False
        success = False
        pass_reward = 0
        if stage_before < stage_after and stage_before < (self._num_curbs + self._num_ceils) and \
                not self._pass_state[stage_before]:
            if (x_after - self._interval_start_pos) / self.dt < self._config["x_vel_limit"] - 1:
                done = True
                print('slow')
            self._interval_start_pos = x_after
            self._interval_time = 0

            pass_reward = self._config["pass_reward"]
            self._pass_state[stage_before] = True
            success = True
            self._success_count += 1
            print('success obstacle {} times'.format(self._success_count))
            if self._success_count == 5:
                done = True
                print('Done (success {} times)'.format(self._success_count))

        collision_penalty = 0
        if self.collision_detection('curb') or self.collision_detection('ceil'):
            collision_penalty = -self._config["collision_penalty"]
            if self._config["done_when_collide"] != 0:
                done = True
                print("Collided")

        near_ceiling = False
        for i in range(self._num_ceils + self._num_curbs):
            pos = self._obstacle_pos[i]
            size = self._obstacle_size[i]
            if not self._obstacle_type[i] and \
                    pos - size - self._config['ob_detect_dist'] < x_after and x_after < pos + size + 3:
                near_ceiling = True

        x_vel_reward = 0
        angle_reward = 0
        height_reward = 0
        alive_reward = 0
        foot_reward = 0
        jump_reward = 0
        ctrl_reward = self._ctrl_reward(a)

        height = self.data.qpos[1]
        angle = self.data.qpos[2]
        delta_h = self.data.body_xpos[1, 2] - max(self.data.body_xpos[4, 2], self.data.body_xpos[7, 2])
        nz = np.cos(self.data.qpos[2])
        x_vel = (x_after - x_before) / self.dt
        x_vel = self._config["x_vel_limit"] - abs(x_vel - self._config["x_vel_limit"])
        y_vel = (y_after - y_before) / self.dt
        y_vel = np.clip(y_vel, -self._config["y_vel_limit"], self._config["y_vel_limit"])
        right_foot_vel = abs(right_foot_after - right_foot_before) / self.dt
        left_foot_vel = abs(left_foot_after - left_foot_before) / self.dt

        x_vel_reward = self._config["x_vel_reward"] * x_vel
        angle_reward = self._config["angle_reward"] * nz
        alive_reward = self._config["alive_reward"]
        if near_ceiling:
            height_reward = -self._config["height_reward"] * abs(1.1 - delta_h)
        foot_reward = -self._config["foot_reward"] * (right_foot_vel + left_foot_vel)

        if self._obstacle_type[self._stage]:
            c_pos = self._obstacle_pos[self._stage]
            c_size = self._obstacle_size[self._stage]
            for i, x in enumerate([c_pos - c_size, c_pos, c_pos + c_size]):
                if foot_before <= x and x < foot_after:
                    pass_reward += self._config["pass_reward"] * (i + 1) / 3
                if x_before <= x and x < x_after:
                    jump_reward += self._config["jump_reward"] * y_vel

        if self._config["sparse_reward"] == 0:
            reward = alive_reward + ctrl_reward + x_vel_reward + angle_reward + \
                height_reward + pass_reward + collision_penalty + foot_reward + jump_reward
        else:
            reward = float(success)

        if near_ceiling:
            done = done or height < 0.3
        else:
            done = done or height < self._config["min_height"]

        ob = self._get_obs()
        info = {"alive_reward": alive_reward,
                "ctrl_reward": ctrl_reward,
                "collision_penalty": collision_penalty,
                "height_reward": height_reward,
                "angle_reward": angle_reward,
                "x_vel_reward": x_vel_reward,
                "foot_reward": foot_reward,
                "jump_reward": jump_reward,
                "pass_reward": pass_reward,
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
        obstacle_obs = self._get_obstacles_observation()
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10), qacc, obstacle_obs]).ravel()

    def get_ob_dict(self, ob):
        if len(ob.shape) > 1:
            return {
                'joint': ob[:, :17],
                'acc': ob[:, 17:26],
                'curb': ob[:, -2:]
            }
        return {
            'joint': ob[:17],
            'acc': ob[17:26],
            'curb': ob[-2:]
        }

    def reset_model(self):
        self._put_obstacles()
        self._pass_state = [False] * (self._num_curbs + self._num_ceils)
        self._success_count = 0
        self._meta_policy_stage = -1
        self._stage = 0
        self._curb_stage = 0
        self._ceil_stage = 0
        self._interval_time = 0
        self._interval_start_pos = 0

        r = self._config["init_randomness"]
        self.set_state(
            self.init_qpos + np.random.uniform(-r, r, size=self.model.nq),
            self.init_qvel + np.random.uniform(-r, r, size=self.model.nv)
        )

        # more perturb
        for _ in range(int(self._config["random_steps"])):
            self.do_simulation(self.unwrapped.action_space.sample(), self.frame_skip)
        return self._get_obs()

    def _random_obstacles(self):
        idx = np.random.permutation(np.arange(self._num_curbs + self._num_ceils))
        self._obstacle_type = [True] * (self._num_curbs + self._num_ceils + 1)
        for i in idx[self._num_curbs:]:
            self._obstacle_type[i] = False # ceilings

        r = self._config["obs_randomness"]
        obs_x = self._possible_x + np.random.uniform(-r, r, self._num_curbs + self._num_ceils)
        return obs_x

    def _put_obstacles(self):
        geom_name_list = self.model.geom_names
        self._obstacle_pos = []
        self._obstacle_size = []

        obs_x = self._random_obstacles()

        curb_idx = 1
        ceil_idx = 1
        for i in range(self._num_curbs + self._num_ceils):
            if self._obstacle_type[i]:
                idx = self.model.geom_name2id('curb{}'.format(curb_idx))
                self.model.geom_pos[idx][2] = self._config["curb_height"]
                self.model.geom_size[idx][2] = self._config["curb_height"]
                curb_idx += 1
            else:
                idx = self.model.geom_name2id('ceiling{}'.format(ceil_idx))
                self.model.geom_pos[idx][2] = self._config["ceil_height"]
                self.model.geom_size[idx][0] = self._config["ceil_size"]
                ceil_idx += 1
            self.model.geom_pos[idx][0] = obs_x[i]
            self._obstacle_pos.append(self.model.geom_pos[idx][0])
            self._obstacle_size.append(self.model.geom_size[idx][0])
        self._obstacle_pos.append(100)
        self._obstacle_size.append(1)

    def get_next_primitive(self, ob, prev_primitive):
        self._get_obstacles_observation()
        # after successful jump
        if self._stage != self._meta_policy_stage:
            self._meta_policy_stage = self._stage
            return 'forward'
        if self._obstacle_type[self._stage]:
            return 'jump'
        else:
            return 'crawl'
