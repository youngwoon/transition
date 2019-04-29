import os.path as osp
import glob
from collections import deque

import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import h5py

from baselines import logger
from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U

from rl.ops import activation
from rl.util import make_env


class Replay(object):
    def __init__(self, max_size=1e6, name='buffer'):
        self.max_size = max_size
        self.obs = deque(maxlen=max_size)
        self.collected_obs = []
        self.name = name

    def size(self):
        return len(self.obs) + len(self.collected_obs)

    def list(self):
        return list(self.obs) + self.collected_obs

    def add_collected_obs(self, ob):
        if isinstance(ob, list):
            self.collected_obs.extend(ob)
        elif isinstance(ob, np.ndarray):
            if len(ob.shape) == 1:
                self.collected_obs.append(ob)
            else:
                for i in range(ob.shape[0]):
                    self.collected_obs.append(ob[i, :])
        else:
            self.collected_obs.append(np.ones((1,)) * ob)

    def add(self, ob):
        if isinstance(ob, list):
            self.obs.extend(ob)
        elif isinstance(ob, np.ndarray):
            if len(ob.shape) == 1:
                self.obs.append(ob)
            else:
                for i in range(ob.shape[0]):
                    self.obs.append(ob[i, :])
        else:
            self.obs.append(np.ones((1,)) * ob)

    def sample(self, batchsize):
        idx = []
        idx_collected = []
        if len(self.collected_obs) and len(self.obs):
            idx = np.random.randint(len(self.obs), size=batchsize//2)
            idx_collected = np.random.randint(len(self.collected_obs), size=batchsize//2)
            return np.concatenate([self.get(idx), self.get_collected_obs(idx_collected)], axis=0)
        elif len(self.collected_obs):
            idx_collected = np.random.randint(len(self.collected_obs), size=batchsize)
            return self.get_collected_obs(idx_collected)
        else:
            idx = np.random.randint(len(self.obs), size=batchsize)
            return self.get(idx)

    def get(self, idx):
        return np.stack([self.obs[i] for i in idx])

    def get_collected_obs(self, idx):
        return np.stack([self.collected_obs[i] for i in idx])

    def iterate_times(self, batchsize, times):
        for x in range(times):
            yield self.sample(batchsize)


class ProximityPredictor(object):
    def __init__(self, name, path, env, ob_env_name, is_train=True,
                 use_traj_portion_start=0.0, use_traj_portion_end=1.0, config=None):
        self._scope = 'proximity_predictor/' + name
        self.env_name = name.split('.')[0]
        self._config = config

        # make primitive env for observation
        self._env = make_env(ob_env_name, config)
        self._include_acc = config.proximity_include_acc
        self._ob_shape = self._env.unwrapped.ob_shape
        self.ob_type = sorted(self._env.unwrapped.ob_type)
        if not self._include_acc and 'acc' in self.ob_type:
            self._ob_shape.pop('acc')
            self.ob_type.remove('acc')

        self.obs_norm = config.proximity_obs_norm
        self.observation_shape = np.sum([np.prod(ob) for ob in self._ob_shape.values()])

        # replay buffers
        self.fail_buffer = Replay(
            max_size=config.proximity_replay_size, name='fail_buffer')
        self.success_buffer = Replay(
            max_size=config.proximity_replay_size, name='success_buffer')

        # build the architecture
        self._num_hidden_layer = config.proximity_num_hid_layers
        self._hidden_size = config.proximity_hid_size
        self._activation_fn = activation(config.proximity_activation_fn)
        self._build_ph()

        logger.info('===== Proximity_predictor for {} ====='.format(self._scope))
        # load collected states
        if is_train or config.evaluate_proximity_predictor:
            state_file_path = osp.join(config.primitive_dir, path.split('/')[0], 'state')
            logger.info('Search state files from: {}'.format(config.primitive_dir))
            state_file_list = glob.glob(osp.join(state_file_path, '*.hdf5'))
            logger.info('Candidate state files: {}'.format(
                ' '.join([f.split('/')[-1] for f in state_file_list])))
            state_file = {}
            try:
                logger.info('Use state files: {}'.format(state_file_list[0].split('/')[-1]))
                state_file = h5py.File(state_file_list[0], 'r')
            except:
                logger.warn("No collected state hdf5 file is located at {}".format(
                    state_file_path))
            logger.info('Use traj portion: {} to {}'.format(
                use_traj_portion_start, use_traj_portion_end))

            if self._config.proximity_keep_collected_obs:
                add_obs = self.success_buffer.add_collected_obs
            else:
                add_obs = self.success_buffer.add

            for k in list(state_file.keys()):
                traj_state = state_file[k]['obs'].value
                start_idx = int(traj_state.shape[0]*use_traj_portion_start)
                end_idx = int(traj_state.shape[0]*use_traj_portion_end)
                try:
                    if state_file[k]['success'].value == 1:
                        traj_state = traj_state[start_idx:end_idx]
                    else:
                        continue
                except:
                    traj_state = traj_state[start_idx:end_idx]
                for t in range(traj_state.shape[0]):
                    ob = traj_state[t][:self.observation_shape]
                    # [ob, label]
                    add_obs(np.concatenate((ob, [1.0]), axis=0))

            # shape [num_state, dim_state]
            logger.info('Size of collected state: {}'.format(self.success_buffer.size()))
            logger.info('Average of collected state: {}'.format(np.mean(self.success_buffer.list(), axis=0)))

        # build graph
        fail_logits, fail_target_value, success_logits, success_target_value = \
            self._build_graph(self.fail_obs_ph, self.success_obs_ph, reuse=False)

        # compute prob
        fake_prob = tf.reduce_mean(fail_logits)  # should go to 0
        real_prob = tf.reduce_mean(success_logits)  # should go to 1

        # compute loss
        if config.proximity_loss_type == 'lsgan':
            self.fake_loss = tf.reduce_mean((fail_logits - fail_target_value)**2)
            self.real_loss = tf.reduce_mean((success_logits - success_target_value)**2)
        elif config.proximity_loss_type == 'wgan':
            self.fake_loss = tf.reduce_mean(tf.abs(fail_logits - fail_target_value))
            self.real_loss = tf.reduce_mean(tf.abs(success_logits - success_target_value))

        # loss + accuracy terms
        self.total_loss = self.fake_loss + self.real_loss
        self.losses = {
            "fake_loss": self.fake_loss,
            "real_loss": self.real_loss,
            "fake_prob": fake_prob,
            "real_prob": real_prob,
            "total_loss": self.total_loss
        }

        # predict proximity
        self._proximity_op = tf.clip_by_value(success_logits, 0, 1)[:, 0]

    def _build_ph(self):
        self.fail_obs_ph = tf.placeholder(
            dtype=tf.float32, name="fail_obs_ph",
            shape=[None, self.observation_shape+1])
        self.success_obs_ph = tf.placeholder(
            dtype=tf.float32, name="success_obs_ph",
            shape=[None, self.observation_shape+1])

    def _model(self, obs, reuse=False):
        with tf.variable_scope(self._scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            _ = obs

            if self.obs_norm:
                # obs normalization
                with tf.variable_scope("ob_rms"):
                    self.ob_rms = RunningMeanStd(shape=self.observation_shape)
                _ = (_ - self.ob_rms.mean) / self.ob_rms.std
                _ = tf.clip_by_value(_, -5.0, 5.0)
            else:
                self.ob_rms = None

            for i in range(self._num_hidden_layer):
                _ = layers.fully_connected(
                    _, self._hidden_size, activation_fn=self._activation_fn)
            logits = layers.fully_connected(_, 1, activation_fn=None)
        return logits

    def _build_graph(self, fail_obs_ph, success_obs_ph, reuse=False):
        fail_logits = self._model(fail_obs_ph[:, :-1], reuse=False)
        success_logits = self._model(success_obs_ph[:, :-1], reuse=True)
        fail_target_value = fail_obs_ph[:, -1]
        success_target_value = success_obs_ph[:, -1]
        return fail_logits, fail_target_value, success_logits, success_target_value

    def get_ob_dict(self, ob):
        if not isinstance(ob, dict):
            ob = self._env.env.get_ob_dict(ob)
        ob_dict = {}
        for ob_name in self.ob_type:
            if len(ob[ob_name].shape) == 1:
                t_ob = ob[ob_name][None]
            else:
                t_ob = ob[ob_name]
            ob_dict[ob_name] = t_ob
        return ob_dict

    def get_ob_list(self, ob):
        ob_list = []
        if not isinstance(ob, dict):
            ob = self._env.env.get_ob_dict(ob)
        for ob_name in self.ob_type:
            if len(ob[ob_name].shape) == 1:
                t_ob = ob[ob_name][None]
            else:
                t_ob = ob[ob_name]
            ob_list.append(t_ob)
        return ob_list

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self._scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self._scope)

    def proximity(self, obs):
        sess = U.get_session()
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        if obs.shape[1] == self.observation_shape:
            obs = np.concatenate((obs, np.zeros(shape=[obs.shape[0], 1])), axis=1)
        proximity = sess.run(self._proximity_op, feed_dict={self.success_obs_ph: obs})
        return proximity

    def _sample_batch(self, final_buffer, batchsize):
        assert final_buffer.size() > 0
        return final_buffer.sample(batchsize)

    def sample_fail_batch(self, batchsize):
        return self._sample_batch(self.fail_buffer, batchsize)

    def sample_success_batch(self, batchsize):
        return self._sample_batch(self.success_buffer, batchsize)

    def reset(self):
        with tf.variable_scope(self._scope, reuse=True):
            varlist = self.get_trainable_variables()
            initializer = tf.variables_initializer(varlist)
            U.get_session().run(initializer)

    def evaluate_success_states(self):
        proximity = []
        for _ in range(1000):
            obs = self.success_buffer.sample(100)
            proximity.append(self.proximity(obs))
        proximity = np.concatenate(proximity, axis=0)
        return np.mean(proximity), np.std(proximity)

