import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from gym.envs.mujoco.jaco import JacoEnv


class JacoPickEnv(JacoEnv):
    def __init__(self, with_rot=1):
        super().__init__(with_rot=with_rot)

        # config
        self._config.update({
            "pick_reward": 100,
            "hold_reward": 2,
            "guide_reward": 2,
            "success_reward": 10,
            "hold_reward_duration": 50,
            "random_box": 1,
            "init_randomness": 0.05,
            "random_steps": 10,
        })

        # state
        self._t = 0
        self._hold_duration = 0
        self._picked = False
        self._pick_height = 0
        self._dist_boxtop = 0

        # env info
        self.reward_type += ["guide_reward", "pick_reward", "hold_reward", "success_reward", "success"]
        self.ob_type = self.ob_shape.keys()

        mujoco_env.MujocoEnv.__init__(self, "jaco_pick.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        self._t += 1
        self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        done = False
        success = False
        guide_reward = 0
        pick_reward = 0
        hold_reward = 0
        ctrl_reward = self._ctrl_reward(a)
        success_reward = 0

        hand_pos = self._get_hand_pos()
        box_z = self._get_box_pos()[2]
        dist_hand = self._get_distance_hand('box')
        boxtop = self._get_box_pos()
        #boxtop = self._get_box_pos() + [0, 0, 0.05]
        dist_boxtop = np.linalg.norm(hand_pos - boxtop)
        #in_air = box_z > 0.06
        #in_hand = dist_hand < 0.06
        in_air = box_z > 0.05
        in_hand = dist_hand < 0.06

        # guide hand to top of box
        #if not self._picked and not (in_hand and in_air):
        if not self._picked and not in_hand:
            guide_reward = self._config["guide_reward"] * (self._dist_boxtop - dist_boxtop)
            self._dist_boxtop = dist_boxtop

        # pick
        if in_hand and in_air and self._pick_height < min(self._init_box_pos_above[2], box_z):
            pick_reward = self._config["pick_reward"] * \
                (min(self._init_box_pos_above[2], box_z) - self._pick_height)
            self._pick_height = box_z

        if not self._picked and in_hand and in_air:
            self._picked = True
            pick_reward += 1

        #if self._picked and not (in_hand and in_air):
        if self._picked and not in_hand:
            done = True

        # hold
        if in_hand and in_air:
            dist = np.linalg.norm(self._init_box_pos_above - self._get_box_pos())
            hold_reward = self._config["hold_reward"] * (1 - dist)
            if dist < 0.1:
                self._hold_duration += 1
            if self._config['hold_reward_duration'] == self._hold_duration:
                print('success pick!')
                done = success = True
                success_reward = self._config["success_reward"] * (200 - self._t)

        reward = ctrl_reward + pick_reward + hold_reward + guide_reward + success_reward
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
                'hand': ob[:, 46:49]
            }
        else:
            return {
                'joint': ob[:31],
                'acc': ob[31:46],
                'hand': ob[46:49]
            }

    def reset_box(self):
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()

        # set box's initial position
        self._init_box_pos = np.asarray(
            [0.5 + np.random.uniform(0, 0.1) * self._config["random_box"],
             0.2 + np.random.uniform(0, 0.1) * self._config["random_box"],
             0.03])
        qpos[9:12] = self._init_box_pos

        # set box's initial rotation
        qpos[12:16] = self.init_qpos[12:16] + self.np_random.uniform(low=-.005, high=.005, size=4)
        if self._config['init_random_rot']:
            qpos[12:16] = 0
            qpos[12] = self.np_random.randint(-1, 2)
            qpos[13 + self.np_random.randint(3)] = self.np_random.randint(-1, 2)
            if np.all(qpos[12:16] == 0):
                qpos[12] = 1
            qpos[12:16] += self.np_random.uniform(low=-.005, high=.005, size=4)
        qvel[9:15] = self.init_qvel[9:15] + self.np_random.uniform(low=-.005, high=.005, size=6)
        self.set_state(qpos, qvel)


    def reset_model(self):
        init_randomness = self._config["init_randomness"]
        qpos = self.init_qpos + self.np_random.uniform(low=-init_randomness,
                                                       high=init_randomness,
                                                       size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-init_randomness,
                                                       high=init_randomness,
                                                       size=self.model.nv)
        self.set_state(qpos, qvel)

        self.reset_box()

        # more perturb
        for _ in range(int(self._config["random_steps"])):
            self.do_simulation(self.unwrapped.action_space.sample(), self.frame_skip)

        self._t = 0
        self._hold_duration = 0
        self._pick_height = 0
        self._picked = False
        #boxtop = self._init_box_pos + [0, 0, 0.05]
        boxtop = self._init_box_pos
        self._dist_boxtop = np.linalg.norm(self._get_hand_pos() - boxtop)
        self._init_box_pos_above = self._init_box_pos.copy()
        self._init_box_pos_above[2] = 0.4

        return self._get_obs()

    def is_terminate(self, ob, success_length=50, init=False, env=None):
        if init:
            self.count_evaluate = 0
            self.success = True

        box_pos = ob[9:12]
        hand_pos = ob[46:49]
        dist_hand = np.linalg.norm(box_pos - hand_pos)
        box_z = box_pos[2]
        in_air = box_z > 0.06
        on_ground = box_z < 0.06
        in_hand = dist_hand < 0.06

        if on_ground and self.count_evaluate > 0:
            self.success = False

        if in_air and in_hand:
            self.count_evaluate += 1

        return self.success and self.count_evaluate >= success_length
