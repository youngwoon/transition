import numpy as np

from gym.envs.mujoco import BaseEnv


class JacoEnv(BaseEnv):
    def __init__(self, with_rot=1):
        # config
        self._with_rot = with_rot
        self._config = {
            "random_steps": 0,
            "ctrl_reward": 1e-4,
            "init_random_rot": 0
        }
        self._fail = False

        # env info
        self.reward_type = ["ctrl_reward"]
        self.ob_shape = {"joint": [31], "acc": [15], "hand": [3]}
        if self._with_rot == 0:
            self.ob_shape["joint"] = [24]  # 4 for orientation, 3 for velocity
            self.ob_shape["acc"] = [12]  # 4 for orientation, 3 for velocity

    def _ctrl_reward(self, a):
        ctrl_reward = -self._config["ctrl_reward"] * np.square(a).sum()
        ctrl_reward += -self._config["ctrl_reward"] ** 2 * np.abs(self.data.qvel).mean()
        ctrl_reward += -self._config["ctrl_reward"] ** 2 * np.abs(self.data.qacc).mean()
        return ctrl_reward

    def _get_box_pos(self):
        return self._get_pos('box')

    def _get_target_pos(self):
        return self._get_pos('target')

    def _get_hand_pos(self):
        hand_pos = np.mean([self._get_pos(name) for name in [
            'jaco_link_hand', 'jaco_link_finger_1',
            'jaco_link_finger_2', 'jaco_link_finger_3']], 0)
        return hand_pos

    def _get_distance_hand(self, name):
        pos = self._get_pos(name)
        hand_pos = self._get_hand_pos()
        return np.linalg.norm(pos - hand_pos)

    def viewer_setup(self):
        # self.viewer.cam.trackbodyid = 1
        self.viewer.cam.trackbodyid = -1
        # self.viewer.cam.distance = self.model.stat.extent * 2
        self.viewer.cam.distance = 2.1
        self.viewer.cam.azimuth = 200
        # self.viewer.cam.azimuth = 90
        self.viewer.cam.lookat[0] = 0.5
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.elevation = -20

        #self.viewer.vopt.frame = 1

    def is_terminate(self, ob, init=False, env=None):
        raise NotImplementedError
