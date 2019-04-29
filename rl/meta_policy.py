import tensorflow as tf
import numpy as np

from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.distributions import CategoricalPdType
from baselines.common.atari_wrappers import TransitionEnvWrapper
import baselines.common.tf_util as U

import rl.ops as ops
from rl.util import make_env


class MetaPolicy(object):
    def __init__(self, name, env, ob_env_name, primitives, config):
        # args
        self.name = name
        self._config = config

        # training
        self._hid_size = config.meta_hid_size
        self._num_hid_layers = config.meta_num_hid_layers
        self._activation = ops.activation(config.meta_activation)

        # properties
        primitive_env = make_env(ob_env_name, config)
        self._ob_shape = primitive_env.ob_shape
        self.ob_type = sorted(primitive_env.ob_type)
        if 'acc' in self._ob_shape:
            self._ob_shape.pop('acc')
            self.ob_type.remove('acc')
        primitive_env.close()

        self._env = env
        self._ob_space = np.sum([np.prod(ob) for ob in self._ob_shape.values()])
        self.num_primitives = len(primitives)
        self.primitive_names = primitives

        if not config.meta_oracle:
            self._build()

    def _build(self):
        num_primitives = self.num_primitives
        num_hid_layers = self._num_hid_layers
        hid_size = self._hid_size

        self._obs = {}
        for ob_name, ob_shape in self._ob_shape.items():
            self._obs[ob_name] = U.get_placeholder(
                name="ob_{}".format(ob_name),
                dtype=tf.float32,
                shape=[None] + self._ob_shape[ob_name])
        self._prev_primitive = prev_primitive = U.get_placeholder(name="prev_primitive",
                                                                  dtype=tf.int32,
                                                                  shape=[None])

        with tf.variable_scope(self.name):
            self._scope = tf.get_variable_scope().name

            self.ob_rms = {}
            for ob_name in self.ob_type:
                with tf.variable_scope("ob_rms_{}".format(ob_name)):
                    self.ob_rms[ob_name] = RunningMeanStd(shape=self._ob_shape[ob_name])
            obz = [(self._obs[ob_name] - self.ob_rms[ob_name].mean) / self.ob_rms[ob_name].std
                   for ob_name in self.ob_type]
            obz = [tf.clip_by_value(ob, -5.0, 5.0) for ob in obz]
            obz = tf.concat(obz, -1)

            prev_primitive_one_hot = tf.one_hot(prev_primitive, num_primitives, name="prev_primitive_one_hot")
            obz = tf.concat([obz, prev_primitive_one_hot], -1)

            # value function
            with tf.variable_scope("vf"):
                _ = obz
                for i in range(num_hid_layers):
                    _ = self._activation(
                        tf.layers.dense(
                            _, hid_size, name="fc%d" % (i+1),
                            kernel_initializer=U.normc_initializer(1.0)))
                self.vpred = tf.layers.dense(
                    _, 1, name="vpred",
                    kernel_initializer=U.normc_initializer(1.0))[:, 0]

            # meta policy
            with tf.variable_scope("pol"):
                _ = obz
                for i in range(num_hid_layers):
                    _ = self._activation(
                        tf.layers.dense(
                            _, hid_size, name="fc%i" % (i+1),
                            kernel_initializer=U.normc_initializer(1.0)))
                self.selector = tf.layers.dense(
                    _, num_primitives, name="action",
                    kernel_initializer=U.normc_initializer(0.01))
                self.pdtype = pdtype = CategoricalPdType(num_primitives)
                self.pd = pdtype.pdfromflat(self.selector)

        # sample action
        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self.obs = [self._obs[ob_name] for ob_name in self.ob_type]
        self._act = U.function([stochastic, self._prev_primitive] + self.obs, [ac, self.vpred])

    def act(self, ob, prev_primitive, stochastic):
        ob_list = self.get_ob_list(ob)
        ac, vpred = self._act(stochastic, prev_primitive, *ob_list)
        return ac[0], vpred[0]

    def get_ob_list(self, ob):
        ob_list = []
        if not isinstance(ob, dict):
            ob = self._env.get_ob_dict(ob)
        for ob_name in self.ob_type:
            if len(ob[ob_name].shape) == 1:
                ob_list.append(ob[ob_name][None])
            else:
                ob_list.append(ob[ob_name])
        return ob_list

    def get_variables(self):
        if self._config.meta_oracle:
            return []
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self._scope)

    def get_trainable_variables(self):
        if self._config.meta_oracle:
            return []
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self._scope)

    def reset(self):
        if self._config.meta_oracle:
            return
        with tf.variable_scope(self._scope, reuse=True):
            varlist = self.get_trainable_variables()
            initializer = tf.variables_initializer(varlist)
            U.get_session().run(initializer)
