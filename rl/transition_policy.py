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


class TransitionPolicy(object):
    def __init__(self, name, env, ob_env_name, num_primitives,
                 trans_term_activation='softmax', config=None):
        # configs
        self.term_activation = trans_term_activation
        self._config = config

        # args
        self.name = name
        self.env_name = self.name.split('.')[0]

        # training
        self._hid_size = config.trans_hid_size
        self._num_hid_layers = config.trans_num_hid_layers
        self._gaussian_fixed_var = config.trans_fixed_var
        self._activation = ops.activation(config.trans_activation)
        self._include_acc = config.trans_include_acc

        # properties
        primitive_env = make_env(ob_env_name, config)
        self._ob_shape = primitive_env.ob_shape
        self.ob_type = sorted(primitive_env.ob_type)
        self.primitive_env = primitive_env

        if not self._include_acc and 'acc' in self.ob_type:
            self._ob_shape.pop('acc')
            self.ob_type.remove('acc')

        self._env = env
        self._ob_space = np.sum([np.prod(ob) for ob in self._ob_shape.values()])
        self._ac_space = env.action_space
        self._num_primitives = num_primitives

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
                name="ob_{}".format(ob_name),
                dtype=tf.float32,
                shape=[None] + self._ob_shape[ob_name])
        self._cur_primitive = cur_primitive = \
            U.get_placeholder(name="cur_primitive", dtype=tf.int32, shape=[None])

        # obs normalization
        self.ob_rms = {}
        for ob_name in self.ob_type:
            with tf.variable_scope("ob_rms_{}".format(ob_name)):
                self.ob_rms[ob_name] = RunningMeanStd(shape=self._ob_shape[ob_name])
        obz = [(self._obs[ob_name] - self.ob_rms[ob_name].mean) / self.ob_rms[ob_name].std
               for ob_name in self.ob_type]
        obz = [tf.clip_by_value(ob, -5.0, 5.0) for ob in obz]
        obz = tf.concat(obz, -1)

        cur_primitive_one_hot = tf.one_hot(cur_primitive, self._num_primitives, name="cur_primitive_one_hot")
        obz = tf.concat([obz, cur_primitive_one_hot], -1)

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

            if self.term_activation == 'sigmoid':
                self.term_pred = tf.sigmoid(
                    tf.layers.dense(last_out, 1, name="term_final",
                                    kernel_initializer=U.normc_initializer(1.0))[:, 0])
                stochastic_act = tf.less_equal(
                    (1/(2*self._config.trans_term_prob))*tf.random_uniform(tf.shape(self.term_pred)),
                    self.term_pred)
                determinstic_act = tf.less_equal(
                    (1 - self._config.trans_term_prob) * tf.ones_like(self.term_pred),
                    self.term_pred)
            else:
                self.term_pred = tf.layers.dense(last_out, 2, name="term_final",
                                                    kernel_initializer=U.normc_initializer(0.01))
                self.term_pdtype = term_pdtype = CategoricalPdType(2)
                self.term_pd = term_pdtype.pdfromflat(self.term_pred)
                stochastic_act = self.term_pd.sample()
                determinstic_act = self.term_pd.mode()
        self.pd = pdtype.pdfromflat(pdparam)

        # sample action
        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self.obs = [self._obs[ob_name] for ob_name in self.ob_type]
        term = U.switch(stochastic, stochastic_act, determinstic_act)
        self._act = U.function([stochastic, cur_primitive] + self.obs, [ac, self.vpred, term])
        self._value = U.function([cur_primitive] + self.obs, self.vpred)
        self._term_pred = U.function([stochastic, cur_primitive] + self.obs, self.term_pred)

    def act(self, ob, cur_primitive=None, stochastic=False):
        ob_list = self.get_ob_list(ob)
        ac, vpred, term = self._act(stochastic, cur_primitive, *ob_list)
        return ac[0], vpred[0], term[0]

    def value(self, ob, cur_primitive=None):
        ob_list = self.get_ob_list(ob)
        vpred = self._value(cur_primitive, *ob_list)
        return vpred[0]

    def get_term_pred(self, stochastic, ob, cur_primitive=None):
        ob_list = self.get_ob_list(ob)
        term_pred = self._term_pred(stochastic, cur_primitive, *ob_list)
        return term_pred[0]

    def get_ob_dict(self, ob):
        if not isinstance(ob, dict):
            ob = self._env.get_ob_dict(ob)
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
            ob = self._env.get_ob_dict(ob)
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
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self._scope)
        return var_list

    def reset(self):
        with tf.variable_scope(self._scope, reuse=True):
            varlist = self.get_trainable_variables()
            initializer = tf.variables_initializer(varlist)
            U.get_session().run(initializer)
