import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env


class BaseEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self._config = {}
        self._fail = False

    def set_environment_config(self, config):
        for k, v in config.items():
            assert k in self._config, '{} is not in the self._config'.format(k)
            self._config[k] = v

    def render_frame(self):
        viewer = self._get_viewer()
        self.viewer_setup()
        return viewer._read_pixels_as_in_window()

    def do_simulation(self, a, frame_skip):
        try:
            super().do_simulation(a, frame_skip)
        except:
            print('! warning: simulation is unstable. reset the simulation.')
            self.reset()
            self._fail = True

    def step(self, a):
        raise NotImplementedError

    def _get_obs(self):
        raise NotImplementedError

    # get absolute coordinate
    def _get_pos(self, name):
        if name in self.model.geom_names:
            return self.data.get_geom_xpos(name)
        if name in self.model.body_names:
            return self.data.get_body_xpos(name)
        raise ValueError

    def _set_pos(self, name, pos):
        if name in self.model.geom_names:
            geom_idx = self.model.geom_name2id(name)
            self.model.geom_pos[geom_idx][0:3] = pos[:]
            return
        if name in self.model.body_names:
            body_idx = self.model.body_name2id(name)
            self.model.body_pos[body_idx] = pos[:]
            return
        raise ValueError

    def _get_rot(self, name):
        if name in self.model.body_names:
            return self.data.get_body_xquat(name)
        raise ValueError

    def _get_distance(self, name1, name2):
        pos1 = self._get_pos(name1)
        pos2 = self._get_pos(name2)
        return np.linalg.norm(pos1 - pos2)

    def _get_size(self, name):
        body_idx1 = self.model.body_name2id(name)
        for geom_idx, body_idx2 in enumerate(self.model.geom_bodyid):
            if body_idx1 == body_idx2:
                return self.model.geom_size[geom_idx, :].copy()

    def _set_size(self, name, size):
        body_idx1 = self.model.body_name2id(name)
        for geom_idx, body_idx2 in enumerate(self.model.geom_bodyid):
            if body_idx1 == body_idx2:
                self.model.geom_size[geom_idx, :] = size

    def _get_geom_type(self, name):
        body_idx1 = self.model.body_name2id(name)
        for geom_idx, body_idx2 in enumerate(self.model.geom_bodyid):
            if body_idx1 == body_idx2:
                return self.model.geom_type[geom_idx].copy()

    def _set_geom_type(self, name, geom_type):
        body_idx1 = self.model.body_name2id(name)
        for geom_idx, body_idx2 in enumerate(self.model.geom_bodyid):
            if body_idx1 == body_idx2:
                self.model.geom_type[geom_idx] = geom_type

    def _get_qpos(self, name):
        object_qpos = self.data.get_joint_qpos(name)
        return object_qpos.copy()

    def _set_qpos(self, name, pos, rot=[1, 0, 0, 0]):
        object_qpos = self.data.get_joint_qpos(name)
        assert object_qpos.shape == (7,)
        object_qpos[:3] = pos
        object_qpos[3:] = rot
        self.data.set_joint_qpos(name, object_qpos)

    def _set_color(self, name, color):
        body_idx1 = self.model.body_name2id(name)
        for geom_idx, body_idx2 in enumerate(self.model.geom_bodyid):
            if body_idx1 == body_idx2:
                self.model.geom_rgba[geom_idx, 0:len(color)] = color

    def _mass_center(self):
        mass = np.expand_dims(self.model.body_mass, axis=1)
        xpos = self.data.xipos
        return (np.sum(mass * xpos, 0) / np.sum(mass))

    def _reset_external_force(self):
        xfrc = self.data.xfrc_applied
        for i in range(len(xfrc)):
            xfrc[i, :] = 0

    def _apply_external_force(self, verbose=False):
        xfrc = self.data.xfrc_applied
        force = (np.random.rand(6) * 2 - 1) * self._config['apply_force']
        idx = np.random.randint(len(xfrc))
        xfrc[idx, :] = force

    def collision_detection(self, ref_name=None, body_name=None):
        assert ref_name is not None
        mjcontacts = self.data.contact
        ncon = self.data.ncon
        collision = False
        for i in range(ncon):
            ct = mjcontacts[i]
            g1, g2 = ct.geom1, ct.geom2
            g1 = self.model.geom_names[g1]
            g2 = self.model.geom_names[g2]
            if body_name is not None:
                if (g1.find(ref_name) >= 0 or g2.find(ref_name) >= 0) and \
                    (g1.find(body_name) >= 0 or g2.find(body_name) >= 0):
                    collision = True
                    break
            else:
                if (g1.find(ref_name) >= 0 or g2.find(ref_name) >= 0):
                    collision = True
                    break
        return collision
