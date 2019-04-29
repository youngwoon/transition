import tensorflow as tf
import numpy as np
import gym

from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.distributions import make_pdtype
from baselines.common.distributions import CategoricalPdType
from baselines.common.atari_wrappers import TransitionEnvWrapper
import baselines.common.tf_util as U

import rl.ops as ops
from rl.util import make_env


class PrimitivePolicy(object):
    def __init__(self, name, env, ob_env_name, config=None):
        # configs
        self._config = config

        # args
        self.name = name
        self.env_name = self.name.split('-')[0]

        # training
        self._hid_size = config.primitive_hid_size
        self._num_hid_layers = config.primitive_num_hid_layers
        self._gaussian_fixed_var = config.primitive_fixed_var
        self._activation = ops.activation(config.primitive_activation)
        self._include_acc = config.primitive_include_acc

        # properties
        primitive_env = make_env(ob_env_name, config)
        self.hard_coded = primitive_env.hard_coded
        self._ob_shape = primitive_env.ob_shape
        self.ob_type = sorted(primitive_env.ob_type)

        if not self._include_acc and 'acc' in self.ob_type:
            self._ob_shape.pop('acc')
            self.ob_type.remove('acc')

        self._env = env
        self._ob_space = np.sum([np.prod(ob) for ob in self._ob_shape.values()])
        self._ac_space = primitive_env.action_space

        if config.primitive_use_term:
            self.primitive_env = primitive_env
        else:
            primitive_env.close()

        if not self.hard_coded:
            with tf.variable_scope(self.name):
                self._scope = tf.get_variable_scope().name
                self._build()

    def _build(self):
        ac_space = self._ac_space
        num_hid_layers = self._num_hid_layers
        hid_size = self._hid_size
        gaussian_fixed_var = self._gaussian_fixed_var

        # obs
        self._obs = {}
        for ob_name, ob_shape in self._ob_shape.items():
            self._obs[ob_name] = U.get_placeholder(
                name="ob_{}_primitive".format(ob_name),
                dtype=tf.float32,
                shape=[None] + self._ob_shape[ob_name])

        # obs normalization
        self.ob_rms = {}
        for ob_name in self.ob_type:
            with tf.variable_scope("ob_rms_{}".format(ob_name)):
                self.ob_rms[ob_name] = RunningMeanStd(shape=self._ob_shape[ob_name])
        obz = [(self._obs[ob_name] - self.ob_rms[ob_name].mean) / self.ob_rms[ob_name].std
               for ob_name in self.ob_type]
        obz = [tf.clip_by_value(ob, -5.0, 5.0) for ob in obz]
        obz = tf.concat(obz, -1)

        # value function
        with tf.variable_scope("vf"):
            last_out = obz
            for i in range(num_hid_layers):
                last_out = self._activation(
                    tf.layers.dense(last_out, hid_size, name="fc%i" % (i+1),
                                    kernel_initializer=U.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(last_out, 1, name="final",
                                         kernel_initializer=U.normc_initializer(1.0))[:, 0]

        # primitive policy
        self.pdtype = pdtype = make_pdtype(ac_space)
        with tf.variable_scope("pol"):
            last_out = obz
            for i in range(num_hid_layers):
                last_out = self._activation(
                    tf.layers.dense(last_out, hid_size, name="fc%i" % (i+1),
                                    kernel_initializer=U.normc_initializer(1.0)))

            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = tf.layers.dense(last_out, pdtype.param_shape()[0]//2, name="final",
                                       kernel_initializer=U.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2],
                                         initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name="final",
                                          kernel_initializer=U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        # sample action
        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self.obs = [self._obs[ob_name] for ob_name in self.ob_type]
        self._act = U.function([stochastic] + self.obs, [ac, self.vpred])
        self._value = U.function(self.obs, self.vpred)

    def act(self, ob, stochastic):
        if self.hard_coded:
            return self.primitive_env.unwrapped.act(ob), 0
        ob_list = self.get_ob_list(ob)
        ac, vpred = self._act(stochastic, *ob_list)
        return ac[0], vpred[0]

    def value(self, ob):
        if self.hard_coded:
            return 0
        ob_list = self.get_ob_list(ob)
        vpred = self._value(*ob_list)
        return vpred[0]

    def get_ob_dict(self, ob):
        if not isinstance(ob, dict):
            ob = self._env.get_ob_dict(ob)
        ob_dict = {}
        for ob_name in self.ob_type:
            if len(ob[ob_name].shape) == 1:
                t_ob = ob[ob_name][None]
            else:
                t_ob = ob[ob_name]
            t_ob = t_ob[:, -np.sum(self._ob_shape[ob_name]):]
            ob_dict[ob_name] = t_ob
        return ob_dict

    def get_ob_list(self, ob):
        ob_list = []
        if not isinstance(ob, dict):
            ob = self._env.get_ob_dict(ob)
        for ob_name in self.ob_type:
            if len(ob[ob_name].shape) == 1:
                t_ob = ob[ob_name][None]
            else:
                t_ob = ob[ob_name]
            t_ob = t_ob[:, -np.sum(self._ob_shape[ob_name]):]
            ob_list.append(t_ob)
        return ob_list

    def is_terminate(self, ob, init=False, env=None):
        return self.primitive_env.unwrapped.is_terminate(ob, init=init, env=env)

    def get_variables(self):
        if self.hard_coded:
            return []
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self._scope)

    def get_trainable_variables(self):
        if self.hard_coded:
            return []
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self._scope)
        return var_list

    def reset(self):
        if self.hard_coded:
            return
        with tf.variable_scope(self._scope, reuse=True):
            varlist = self.get_trainable_variables()
            initializer = tf.variables_initializer(varlist)
            U.get_session().run(initializer)
