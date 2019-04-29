'''
This code is highly based on https://github.com/carpedm20/deep-rl-tensorflow/blob/master/agents/statistic.py
'''

import tensorflow as tf
import numpy as np

import baselines.common.tf_util as U


class stats():

    def __init__(self, scalar_keys=[], histogram_keys=[]):
        self.scalar_keys = scalar_keys
        self.histogram_keys = histogram_keys
        self.scalar_summaries = []
        self.scalar_summaries_ph = []
        self.histogram_summaries_ph = []
        self.histogram_summaries = []

        self.summaries_map = {}
        scopes = set()
        for key in scalar_keys:
            words = key.split('/')
            if len(words) > 1:
                scopes.add(words[0])

        registered_keys = []
        for scope in scopes:
            with tf.variable_scope(scope):
                for k in scalar_keys:
                    if k.startswith(scope):
                        name = '/'.join(k.split('/')[1:])
                        ph = tf.placeholder('float32', None, name=name)
                        sm = tf.summary.scalar(name, ph)
                        self.scalar_summaries_ph.append(ph)
                        self.scalar_summaries.append(sm)
                        registered_keys.append(k)
                        self.summaries_map[k] = ph

        with tf.variable_scope('summary'):
            for k in scalar_keys:
                if k not in registered_keys:
                    ph = tf.placeholder('float32', None, name=k+'.scalar.summary')
                    sm = tf.summary.scalar(k, ph)
                    self.scalar_summaries_ph.append(ph)
                    self.scalar_summaries.append(sm)
                    self.summaries_map[k] = ph
            for k in histogram_keys:
                ph = tf.placeholder('float32', None, name=k+'.histogram.summary')
                sm = tf.summary.histogram(k, ph)
                self.histogram_summaries_ph.append(ph)
                self.histogram_summaries.append(sm)
                self.summaries_map[k] = ph

        self.summaries = tf.summary.merge(self.scalar_summaries+self.histogram_summaries)

    def add_all_summary(self, writer, values, iter):
        # Note that the order of the incoming ```values``` should be the same as the that of the
        #            ```scalar_keys``` given in ```__init__```
        if np.sum(np.isnan(values)+0) != 0:
            return
        sess = U.get_session()
        keys = self.scalar_summaries_ph + self.histogram_summaries_ph
        feed_dict = {}
        for k, v in zip(keys, values):
            feed_dict.update({k: v})
        summaries_str = sess.run(self.summaries, feed_dict)
        writer.add_summary(summaries_str, iter)

    def add_all_summary_dict(self, writer, value_dict, iter):
        # Note that the order of the incoming ```values``` should be the same as the that of the
        #            ```scalar_keys``` given in ```__init__```
        sess = U.get_session()
        feed_dict = {}
        for k in self.summaries_map.keys():
            if k not in value_dict:
                value_dict[k] = 0
            elif np.sum(np.isnan(value_dict[k])+0) != 0:
                print('[!] NaN for the summary value of {}'.format(k))
                value_dict[k] = 1e30
            feed_dict.update({self.summaries_map[k]: value_dict[k]})
        summaries_str = sess.run(self.summaries, feed_dict)
        writer.add_summary(summaries_str, iter)
