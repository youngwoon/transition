import os.path as osp
import os
from collections import defaultdict

import numpy as np
import tensorflow as tf
from mpi4py import MPI
from tqdm import trange
import moviepy.editor as mpy
import h5py

from baselines.common import zipsame
from baselines import logger
import baselines.common.tf_util as U
from baselines.common.mpi_adam import MpiAdam
from baselines.statistics import stats

import rl.rollouts as rollouts
from rl.dataset import Dataset


class Trainer(object):
    def __init__(self, env, meta_pi, meta_oldpi, proximity_predictors,
                 num_primitives, trans_pis, trans_oldpis, config):
        self._env = env
        self._config = config
        self.meta_pi = meta_pi
        self.meta_oldpi = meta_oldpi
        self.proximity_predictors = proximity_predictors
        self._use_proximity_predictor = config.use_proximity_predictor
        self.trans_pis = trans_pis
        self.trans_oldpis = trans_oldpis
        self._num_primitives = num_primitives
        self._use_trans = config.use_trans

        self._cur_lrmult = 0
        self._entcoeff = config.entcoeff
        self._meta_entcoeff = config.meta_entcoeff
        self._trans_entcoeff = config.trans_entcoeff
        self._optim_epochs = config.optim_epochs
        self._optim_proximity_epochs = config.proximity_optim_epochs
        self._optim_stepsize = config.optim_stepsize
        self._optim_proximity_stepsize = config.proximity_learning_rate
        self._optim_batchsize = config.optim_batchsize

        # global step
        self.global_step = tf.Variable(0, name='global_step', dtype=tf.int64, trainable=False)
        self.update_global_step = tf.assign(self.global_step, self.global_step + 1)

        # tensorboard summary
        self._is_chef = (MPI.COMM_WORLD.Get_rank() == 0)
        if self._is_chef:
            self.summary_name = ["reward", "length"]
            self.summary_name += env.unwrapped.reward_type
            self.summary_histogram_name = ['reward_dist', 'primitive_dist']
            if self._use_trans:
                for pi in self.trans_pis:
                    self.summary_name += ["trans_{}/average_length".format(pi.env_name)]
                    self.summary_name += ["trans_{}/rew".format(pi.env_name)]
                    if self._use_proximity_predictor:
                        self.summary_name += ["trans_{}/proximity_rew".format(pi.env_name)]
                self.summary_histogram_name += ["trans_len_histogram"]
                if self._use_proximity_predictor:
                    for proximity in self.proximity_predictors:
                        self.summary_histogram_name += [
                            'proximity_predictor_{}/hist_success_final'.format(proximity.env_name)]
                        self.summary_histogram_name += [
                            'proximity_predictor_{}/hist_success_intermediate'.format(proximity.env_name)]
                        self.summary_histogram_name += [
                            'proximity_predictor_{}/hist_fail_final'.format(proximity.env_name)]
                        self.summary_histogram_name += [
                            'proximity_predictor_{}/hist_fail_intermediate'.format(proximity.env_name)]

        # build loss/optimizers
        self._build()

        if self._is_chef and self._config.is_train:
            self.ep_stats = stats(self.summary_name, self.summary_histogram_name)
            self.writer = U.file_writer(config.log_dir)

    def _build(self):
        config = self._config
        meta_pi = self.meta_pi
        meta_oldpi = self.meta_oldpi
        proximity_predictors = self.proximity_predictors
        trans_pis = self.trans_pis
        trans_oldpis = self.trans_oldpis

        # input placeholders
        atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
        ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

        lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])  # learning rate multiplier, updated with schedule
        self._clip_param = config.clip_param * lrmult  # Annealed cliping parameter epislon

        # meta policy
        if not config.meta_oracle:
            ac = meta_pi.pdtype.sample_placeholder([None])
            prev_primitive = U.get_placeholder_cached(name="prev_primitive")

            var_list = meta_pi.get_trainable_variables()
            self.meta_adam = MpiAdam(var_list)
            fetch_dict = self.policy_loss_ppo(meta_pi, meta_oldpi, ac, atarg, ret,
                                            entcoeff=self._meta_entcoeff)
            if self._is_chef:
                self.summary_name += ['meta/' + key for key in fetch_dict.keys()]
                self.summary_name += ['meta/grad_norm', 'meta/grad_norm_clipped', 'meta/global_norm']
            fetch_dict['g'] = U.flatgrad(fetch_dict['total_loss'], var_list, clip_norm=config.meta_max_grad_norm)
            self.meta_loss = U.function([lrmult, prev_primitive, ac, atarg, ret] + meta_pi.obs, fetch_dict)
            self._update_old_meta = U.function([], [], updates=[
                tf.assign(oldv, newv) for (oldv, newv) in zipsame(
                    meta_oldpi.get_variables(), meta_pi.get_variables())])
            self.zerograd_meta = U.function([], self.nograd(var_list))
            self._meta_global_norm = U.function([], tf.global_norm([tf.cast(var, tf.float32) for var in meta_pi.get_variables()]))

        # proximity_predictor
        if self._use_proximity_predictor:
            self._proximity_predictor_adams = []
            self._proximity_predictor_losses = []
            self._proximity_predictor_zerograds = []
            for i in range(len(proximity_predictors)):
                proximity_predictor = proximity_predictors[i]
                var_list = proximity_predictor.get_trainable_variables()
                self._proximity_predictor_adams.append(MpiAdam(var_list, beta1=0.5))
                # get losses
                fetch_dict = proximity_predictor.losses
                # add summary
                if self._is_chef:
                    proximity_name = 'proximity_predictor_{}/'.format(proximity_predictor.env_name)
                    self.summary_name += [proximity_name + key for key in fetch_dict.keys()]
                    self.summary_name += [proximity_name + 'buffer_size_success_final',
                                          proximity_name + 'buffer_size_success_intermediate',
                                          proximity_name + 'buffer_size_fail_final',
                                          proximity_name + 'buffer_size_fail_intermediate',
                                          proximity_name + 'grad_norm',
                                          proximity_name + 'grad_norm_clipped']
                # get gradients
                fetch_dict['g'] = U.flatgrad(fetch_dict['total_loss'], var_list,
                                             clip_norm=config.proximity_max_grad_norm)
                # update
                self._proximity_predictor_losses.append(
                    U.function([proximity_predictor.fail_obs_ph, proximity_predictor.success_obs_ph], fetch_dict))
                self._proximity_predictor_zerograds.append(U.function([], self.nograd(var_list)))
            all_var_list = []
            for proximity_predictor in proximity_predictors:
                all_var_list.extend(proximity_predictor.get_variables())
            self._proximity_predictor_global_norm = U.function([], tf.global_norm([tf.cast(var, tf.float32) for var in all_var_list]))

        # trans policy
        if self._use_trans:
            cur_primitive = U.get_placeholder_cached(name="cur_primitive")
            sp_ac = trans_pis[0].pdtype.sample_placeholder([None])
            term = None
            if trans_pis[0].term_activation == 'sigmoid':
                term = tf.placeholder(dtype=tf.float32, shape=[None])
            else:
                term = trans_pis[0].term_pdtype.sample_placeholder([None])
            self._update_old_trans = []
            self.trans_adams = []
            self.trans_losses = []
            self.trans_zerograds = []
            for i in range(len(trans_pis)):
                trans_pi = trans_pis[i]
                trans_oldpi = trans_oldpis[i]
                var_list = trans_pi.get_trainable_variables()
                self.trans_adams.append(MpiAdam(var_list))
                fetch_dict = self.policy_loss_ppo(trans_pi, trans_oldpi, sp_ac, atarg, ret,
                                                  term, entcoeff=self._trans_entcoeff)
                if self._is_chef:
                    self.summary_name += ['trans_{}/'.format(trans_pi.env_name) + key for key in fetch_dict.keys()]
                    self.summary_name += ['trans_{}/grad_norm'.format(trans_pi.env_name),
                                          'trans_{}/grad_norm_clipped'.format(trans_pi.env_name),
                                          'trans_{}/global_norm'.format(trans_pi.env_name)]
                fetch_dict['g'] = U.flatgrad(fetch_dict['total_loss'], var_list,
                                             clip_norm=config.trans_max_grad_norm)
                self.trans_losses.append(U.function([lrmult, cur_primitive, sp_ac, atarg, ret, term] + trans_pi.obs, fetch_dict))
                self._update_old_trans.append(U.function([], [], updates=[
                    tf.assign(oldv, newv) for (oldv, newv) in zipsame(
                        trans_oldpi.get_variables(), trans_pi.get_variables())]))
                self.trans_zerograds.append(U.function([], self.nograd(var_list)))
            all_var_list = []
            for pi in trans_pis:
                all_var_list.extend(pi.get_variables())
            self._trans_global_norm = U.function([], tf.global_norm([tf.cast(var, tf.float32) for var in all_var_list]))

        # initialize and sync
        U.initialize()
        if not config.meta_oracle:
            self.meta_adam.sync()
        if self._use_proximity_predictor:
            for adam in self._proximity_predictor_adams:
                adam.sync()
        if self._use_trans:
            for adam in self.trans_adams:
                adam.sync()

    def save_buffers(self, ckpt_path):
        buffer_path = ckpt_path + '.hdf5'
        with h5py.File(buffer_path, 'w') as buffer_file:
            for p in self.proximity_predictors:
                grp = buffer_file.create_group(p.env_name)
                grp['success'] = np.asarray(p.success_buffer.obs)
                grp['fail'] = np.asarray(p.fail_buffer.obs)
                logger.info('Store buffers for {}. success states ({})  fail states ({})'.format(
                    p.env_name, grp['success'].shape[0], grp['fail'].shape[0]))

    def summary(self, it):
        if self._is_chef:
            if it % self._config.ckpt_save_step == 0:
                fname = osp.join(self._config.log_dir, '%.5d' % it)
                U.save_state(fname)
                self.save_buffers(fname)

    def train(self, rollout):
        config = self._config
        sess = U.get_session()
        global_step = sess.run(self.global_step)
        t = trange(global_step, config.max_iters,
                   total=config.max_iters, initial=global_step)
        meta_info, trans_info = None, None

        for step in t:
            # backup checkpoint
            self.summary(step)

            if config.lr_decay:
                self._cur_lrmult = max(1.0 - float(step) / config.max_iters, 0)
            else:
                self._cur_lrmult = 1.0
            self._cur_proximity_lrmult = 1.0

            # rollout
            rolls = rollout.__next__()
            rollouts.add_advantage_meta(rolls, 0.99, 0.98, config.meta_duration)

            if self._use_trans:
                trans_seg = rollouts.prepare_all_rolls(
                    rolls, 0.99, 0.98,
                    num_primitives=self._num_primitives, use_trans=self._use_trans,
                    trans_use_task_reward=not self._use_proximity_predictor,
                    config=config)

            # train meta policy
            if not config.meta_oracle:
                num_meta_batches = config.num_rollouts // config.meta_duration // self._optim_batchsize + 1
                meta_info = self._update_meta_policy(rolls, num_meta_batches)

            # train proximity_predictor
            if self._use_proximity_predictor:
                num_proximity_predictor_batches = config.num_rollouts // self._optim_batchsize + 1
                proximity_predictor_info = self._update_proximity_predictor_network(
                    trans_seg, num_proximity_predictor_batches,
                    only_use_trans_term_state=config.proximity_only_use_trans_term_state)

            # train transition policy
            if self._use_trans:
                num_trans_batches = config.num_rollouts // self._optim_batchsize + 1
                trans_info = self._update_trans_policies(trans_seg, num_trans_batches)
                if self._is_chef and trans_info is not None:
                    trans_info['trans_len_histogram'] = rolls["trans_len"]

            # log
            if self._is_chef:
                ep = len(rolls["ep_length"])
                reward_mean = np.mean(rolls["ep_reward"])
                reward_std = np.std(rolls["ep_reward"])
                length_mean = np.mean(rolls["ep_length"])
                length_std = np.std(rolls["ep_length"])
                success_mean = np.mean(rolls["ep_success"])
                success_std = np.std(rolls["ep_success"])
                desc = "ep(%d) reward(%.1f, %.1f) length(%d, %.1f) success(%.1f, %.1f)\n" % (
                    ep, reward_mean, reward_std, length_mean, length_std, success_mean, success_std)

                # TB
                if step % config.write_summary_step == 0:
                    info = {}
                    info['reward_dist'] = rolls["ep_reward"]
                    info['primitive_dist'] = rolls["meta_ac"]
                    for key, value in rolls.items():
                        if key.startswith('ep_'):
                            info[key.split('ep_')[1]] = np.mean(value)

                    if config.meta_oracle is False:
                        info.update(meta_info)
                    if self._use_proximity_predictor:
                        info.update(proximity_predictor_info)
                    if self._use_trans:
                        info.update(trans_info)
                        for i in range(len(self.trans_pis)):
                            trans_name = self.trans_pis[i].env_name
                            idx = (rolls['cur_primitive'] == i) * rolls['is_trans']
                            trans_length = np.sum(rolls['trans_len'][rolls['meta_ac'] == i])
                            trans_num = np.sum((rolls['meta_ac'] == i) * (rolls['trans_len'] > 0))
                            avg_len = trans_length / trans_num if trans_num > 0 else 0
                            info['trans_'+trans_name+'/average_length'] = avg_len
                            if trans_num > 0:
                                if self._use_proximity_predictor:
                                    info['trans_'+trans_name+'/proximity_rew'] = np.sum(rolls["rew"][idx * rolls['term'] == 1]) / trans_num
                                info['trans_'+trans_name+'/rew'] = np.sum(rolls["env_rew"][idx])
                            else:
                                if self._use_proximity_predictor:
                                    info['trans_'+trans_name+'/proximity_rew'] = 0
                                info['trans_'+trans_name+'/rew'] = 0
                            desc += "[%s] trans_len(avg: %.3f, total: %d)\n" % (
                                trans_name, avg_len, trans_length)

                    self.ep_stats.add_all_summary_dict(self.writer, info, global_step)
                t.set_description(desc)
                global_step = sess.run(self.update_global_step)

    def evaluate(self, rollout, ckpt_num=None):
        config = self._config

        ep_lens = []
        ep_rets = []
        ep_success = []
        if config.record:
            record_dir = osp.join(config.log_dir, 'video')
            if config.test_module_net:
                record_dir = osp.join(config.log_dir, 'no_video')
            os.makedirs(record_dir, exist_ok=True)

        if config.proximity_hist:
            proximity_hist = []
            for proximity_idx in range(len(self.proximity_predictors)):
                proximity_hist.append([])

        for _ in range(config.num_evaluation_run):
            ep_traj = rollout.__next__()
            ep_lens.append(ep_traj["ep_length"][0])
            ep_rets.append(ep_traj["ep_reward"][0])
            ep_success.append(np.sum(ep_traj["ep_success"]))
            if config.evaluation_log:
                logger.log('Trial #{}: lengths {}, returns {}'.format(
                    _, ep_traj["ep_length"][0], ep_traj["ep_reward"][0]))

            if config.proximity_hist:
                for proximity_idx in range(len(self.proximity_predictors)):
                    idx = np.where((ep_traj['term'] == 1) * (ep_traj['cur_primitive'] == proximity_idx))
                    proximity_rew = ep_traj['rew'][idx]
                    success = ep_traj['success'][idx]
                    for i in range(proximity_rew.shape[0]):
                        proximity_hist[proximity_idx].append((proximity_rew[i], success[i]))

            # Video recording
            if config.record:
                visual_obs = ep_traj["visual_obs"]
                video_name = (config.video_prefix or '') + '{}{}_rew_{:.2f}_len_{}.mp4'.format(
                    '' if ckpt_num is None else 'ckpt_{}_'.format(ckpt_num), _,
                    ep_traj["ep_reward"][0], ep_traj["ep_length"][0])
                video_path = osp.join(record_dir, video_name)
                fps = 60.

                def f(t):
                    frame_length = len(visual_obs)
                    new_fps = 1./(1./fps + 1./frame_length)
                    idx = min(int(t*new_fps), frame_length-1)
                    return visual_obs[idx]
                video = mpy.VideoClip(f, duration=len(visual_obs)/fps+2)
                video.write_videofile(video_path, fps, verbose=False, progress_bar=False)

        if config.proximity_hist:
            for i in range(len(self.proximity_predictors)):
                proximity_hist[i] = np.array(proximity_hist[i])

        logger.log('Episode Length: {}'.format(sum(ep_lens) / config.num_evaluation_run))
        logger.log('Episode Rewards: {}'.format(sum(ep_rets) / config.num_evaluation_run))
        logger.log('Episode Success: {}'.format(sum(ep_success) / config.num_evaluation_run))

        if config.final_eval:
            file_name = self._config.prefix + ".txt"
            if self._config.test_module_net:
                file_name = 'no_' + file_name
            path = os.path.join(os.path.expanduser('~'), 'iclr_eval', file_name)
            with open(path, "w") as f:
                for s in ep_success:
                    f.write(str(s))
                    f.write('\n')

    def _evaluate_proximity_predictor(self):
        info = {}
        for proximity in self.proximity_predictors:
            logger.log('*** evaluate {}'.format(proximity.env_name))
            mean, std = proximity.evaluate_success_states()
            logger.log('mean: {} std: {}'.format(mean, std))
            info[proximity.env_name] = {'mean': mean, 'std': std}
        return info

    def evaluate_proximity_predictor(self, var_list):
        config = self._config

        if config.evaluate_all_ckpts:
            from glob import glob
            import pandas as pd
            from tqdm import tqdm

            files = glob(os.path.join(config.log_dir, "*.index"))
            files.sort()
            max_step = max([int(os.path.basename(f).split('.')[0]) for f in files])

            results = {}
            for proximity in self.proximity_predictors:
                results[proximity.env_name] = {'mean': [], 'std': [], 'step': []}
            for i in tqdm(range(0, max_step, 25)):
                logger.log('*** evaluate ckpt {}'.format(i))
                U.load_state(os.path.join(config.log_dir, '%.5d' % i), var_list)
                info = self._evaluate_proximity_predictor()
                for proximity_name, proximity_info in info.items():
                    for key, value in proximity_info.items():
                        results[proximity_name][key].append(value)
                    results[proximity_name]['step'].append(i)
            df = pd.DataFrame(results)
            df.to_pickle('proximity_predictor_evaluation.pkl')
        else:
            self._evaluate_proximity_predictor()

    def nograd(self, var_list):
        return tf.concat(axis=0, values=[
            tf.reshape(tf.zeros_like(v), [U.numel(v)])
            for v in var_list
        ])

    def policy_loss_ppo(self, pi, oldpi, ac, atarg, ret, term=None, entcoeff=None):
        kl_oldnew = oldpi.pd.kl(pi.pd)
        ent = pi.pd.entropy()
        mean_kl = U.mean(kl_oldnew)
        mean_ent = U.mean(ent)
        entcoeff = self._entcoeff if entcoeff is None else entcoeff
        logger.info('Policy {} entropy {}'.format(pi.name, entcoeff))
        pol_entpen = -entcoeff * mean_ent

        action_prob = pi.pd.logp(ac) - oldpi.pd.logp(ac)
        action_prob = tf.check_numerics(action_prob, 'check action_prob')
        action_loss = tf.check_numerics(atarg, 'check atarg')
        action_loss = tf.exp(action_prob) * atarg
        action_loss = tf.check_numerics(action_loss, 'check action_loss')

        term_loss = None
        if term is not None:
            # ignore prob of actions if term is True
            action_prob = (1 - tf.to_float(term)) * action_prob
            if pi.term_activation == 'sigmoid':
                term_prob = tf.log(pi.term_pred + 1e-5) - tf.clip_by_value(tf.log(oldpi.term_pred + 1e-5), -20, 20)
            else:
                term_prob = pi.term_pd.logp(term) - tf.clip_by_value(oldpi.term_pd.logp(term), -20, 20)
            action_prob += term_prob
            term_loss = tf.exp(term_prob) * atarg
        ratio = tf.exp(action_prob)

        surr1 = ratio * atarg
        surr2 = U.clip(ratio, 1.0 - self._clip_param, 1.0 + self._clip_param) * atarg
        pol_surr = -U.mean(tf.minimum(surr1, surr2))
        vf_loss = U.mean(tf.square(pi.vpred - ret))
        pol_surr = tf.check_numerics(pol_surr, 'check pol_surr')
        vf_loss = tf.check_numerics(vf_loss, 'check vf_loss')
        total_loss = pol_surr + pol_entpen + vf_loss

        total_loss = tf.check_numerics(total_loss, 'check total_loss')
        losses = {'total_loss': total_loss,
                  'action_loss': action_loss,
                  'pol_surr': pol_surr,
                  'pol_entpen': pol_entpen,
                  'kl': mean_kl,
                  'entropy': mean_ent,
                  'vf_loss': vf_loss}
        if term_loss is not None:
            losses.update({'term_loss': term_loss})
        return losses

    def policy_loss_ppo_term(self, pi, oldpi, atarg, ret, term):
        if pi.term_type == 'sigmoid':
            term_prob = tf.log(pi.term_pred + 1e-5) - tf.clip_by_value(tf.log(oldpi.term_pred + 1e-5), -20, 20)
        else:
            term_prob = pi.term_pd.logp(term) - tf.clip_by_value(oldpi.term_pd.logp(term), -20, 20)
        term_loss = tf.exp(term_prob) * atarg
        ratio = tf.exp(term_prob)

        surr1 = ratio * atarg
        surr2 = U.clip(ratio, 1.0 - self._clip_param, 1.0 + self._clip_param) * atarg
        pol_surr = -U.mean(tf.minimum(surr1, surr2))
        vf_loss = U.mean(tf.square(pi.vpred - ret))
        pol_surr = tf.check_numerics(pol_surr, 'check pol_surr')
        vf_loss = tf.check_numerics(vf_loss, 'check vf_loss')
        total_loss = pol_surr + vf_loss

        total_loss = tf.check_numerics(total_loss, 'check total_loss')
        losses = {'total_loss': total_loss,
                  'pol_surr': pol_surr,
                  'vf_loss': vf_loss,
                  'term_loss': term_loss}
        return losses

    def policy_loss_trpo(self, pi, oldpi, ob, ac, atarg, ret):
        raise NotImplementedError()
        kl_oldnew = oldpi.pd.kl(pi.pd)
        ent = pi.pd.entropy()
        mean_kl = U.mean(kl_oldnew)
        mean_ent = U.mean(ent)
        pol_entpen = -self._entcoeff * mean_ent

        vf_loss = U.mean(tf.square(pi.vpred - ret))

        ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))
        pol_surr = U.mean(ratio * atarg)
        pol_loss = pol_surr + pol_entpen

        losses = {'pol_loss': pol_loss,
                  'pol_surr': pol_surr,
                  'pol_entpen': pol_entpen,
                  'kl': mean_kl,
                  'entropy': mean_ent,
                  'vf_loss': vf_loss}
        return losses

    def _update_meta_policy(self, seg, num_batches):
        ob, prev_primitive, ac, atarg, tdlamret = seg["meta_ob"], seg["meta_prev_primitive"], \
            seg["meta_ac"], seg["meta_adv"], seg["meta_tdlamret"]
        if self._is_chef:
            info = defaultdict(list)

        optim_batchsize = min(self._optim_batchsize, ob.shape[0])
        logger.log("\nOptimizing meta... {} epochs * {} batches * {} batchsize <- {} data".format(
            self._optim_epochs, num_batches, optim_batchsize, ob.shape[0]))

        # normalize advantage
        atarg = (atarg - atarg.mean()) / max(atarg.std(), 0.000001)

        # prepare batches
        d = Dataset(dict(ob=ob, prev_primitive=prev_primitive, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=True)

        pi = self.meta_pi

        ob_dict = self._env.get_ob_dict(ob)
        for ob_name in pi.ob_type:
            pi.ob_rms[ob_name].update(ob_dict[ob_name])

        self._update_old_meta()
        b = 0
        for _ in range(self._optim_epochs):
            # for batch in d.iterate_times(optim_batchsize, num_batches):
            for batch in d.iterate_once(optim_batchsize):
                if b == self._optim_epochs * num_batches:
                    break
                ob_list = pi.get_ob_list(batch["ob"])
                fetched = self.meta_loss(
                    self._cur_lrmult, batch["prev_primitive"],
                    batch["ac"], batch["atarg"], batch["vtarg"], *ob_list)
                self.meta_adam.update(fetched['g'], self._optim_stepsize * self._cur_lrmult)
                b += 1
                if self._is_chef:
                    for key, value in fetched.items():
                        if key != 'g':
                            if np.isscalar(value):
                                info['meta/' + key].append(value)
                            else:
                                info['meta/' + key].extend(value)
                        else:
                            # info['meta/grad_norm'].append(np.linalg.norm(value))
                            grad_norm_value = np.linalg.norm(value)
                            info['meta/grad_norm'].append(grad_norm_value)
                            info['meta/grad_norm_clipped'].append(np.clip(
                                grad_norm_value, 0, self._config.trans_max_grad_norm))
        blank = self.zerograd_meta()
        for _ in range(self._optim_epochs * num_batches - b):
            self.meta_adam.update(blank, self._optim_stepsize * self._cur_lrmult)

        if self._is_chef:
            for key, value in info.items():
                info[key] = np.mean(value)
            info['meta/global_norm'] = self._meta_global_norm()
            info['meta/global_norm_clipped'] = np.clip(
                self._meta_global_norm(), 0, self._config.trans_max_grad_norm)
            return info
        return None

    def _update_one_proximity_predictor_network(self, seg, proximity, adam, loss,
                                                zero_grad, num_batches, idx,
                                                only_use_trans_term_state=False):
        ob = seg["ob"]
        cur_primitive = seg["cur_primitive"]
        success = seg['success']

        logger.log("\033[93m"+"proximity_predictor_{}".format(proximity.env_name)+"\033[0m")

        # remove the info for meta task from ob
        # Assumption: info for meta task is always appended in the end of ob
        # use the states that the match proximity_predictor idx
        if len(cur_primitive == idx) > 0:
            ob = ob[:, :proximity.observation_shape]
            ob_success_final = ob[((cur_primitive == idx) * success * seg['term']) == 1]
            ob_success_intermediate = ob[((cur_primitive == idx) * success * (1 - seg['term'])) == 1]
            ob_fail_final = ob[((cur_primitive == idx) * (1 - success) * seg['term']) == 1]
            ob_fail_intermediate = ob[((cur_primitive == idx) * (1 - success) * (1 - seg['term'])) == 1]

            final_state = seg['term'][(cur_primitive == idx) * success == 1]
            for i in range(final_state.shape[0] - 1, -1, -1):
                if final_state[i] != 1:
                    final_state[i] = final_state[i + 1] + 1
            final_state = final_state[final_state != 1]
            final_state = final_state - 1
            if self._config.proximity_weight_decay_linear:
                ob_success_intermediate_weight = (self._config.trans_duration - final_state) / self._config.trans_duration
            else:
                ob_success_intermediate_weight = self._config.proximity_weight_decay_rate ** final_state
            ob_success_intermediate_weight = ob_success_intermediate_weight.reshape((ob_success_intermediate.shape[0], 1))

            # proximity hist
            rew_success_final = proximity.proximity(ob_success_final)
            rew_success_intermediate = proximity.proximity(ob_success_intermediate)
            rew_fail_final = proximity.proximity(ob_fail_final)
            rew_fail_intermediate = proximity.proximity(ob_fail_intermediate)

            logger.log("    ob_success (final {}, intermediate {})  ob_fail (final {}, intermediate {})".format(
                ob_success_final.shape[0], ob_success_intermediate.shape[0], ob_fail_final.shape[0], ob_fail_intermediate.shape[0]))

            # add [obs, label]
            proximity.fail_buffer.add(np.concatenate((ob_fail_final, np.zeros(
                shape=[ob_fail_final.shape[0], 1])), axis=1))
            proximity.success_buffer.add(np.concatenate((ob_success_final, np.ones(
                shape=[ob_success_final.shape[0], 1])), axis=1))
            proximity.fail_buffer.add(np.concatenate((ob_fail_intermediate, np.zeros(
                shape=[ob_fail_intermediate.shape[0], 1])), axis=1))
            proximity.success_buffer.add(np.concatenate((ob_success_intermediate, np.ones(
                shape=[ob_success_intermediate.shape[0], 1])*ob_success_intermediate_weight), axis=1))

            if proximity.ob_rms:
                proximity.ob_rms.update(ob)
        else:
            ob_success_final = np.zeros(shape=(0, 0))
            ob_success_intermediate = np.zeros(shape=(0, 0))
            ob_fail_final = np.zeros(shape=(0, 0))
            ob_fail_intermediate = np.zeros(shape=(0, 0))
            if proximity.ob_rms:
                proximity.ob_rms.noupdate()

        num_state = ob.shape[0]
        optim_batchsize = self._optim_batchsize
        if 0 in [proximity.fail_buffer.size(), proximity.success_buffer.size()]:
            logger.warn('[!] No transition is used. So the proximity_predictor is not trained.')
            blank = zero_grad()
            for _ in range(self._optim_proximity_epochs * num_batches):
                adam.update(blank, self._optim_proximity_stepsize * self._cur_proximity_lrmult)
            return None

        logger.log("Optimizing proximity_predictor_{}... {} epochs * {} batches * {} batchsize <- {}/{} data".format(
            proximity.env_name, self._optim_proximity_epochs, num_batches,
            optim_batchsize, ob.shape[0], num_state))

        if self._is_chef:
            info = defaultdict(list)

        # update proximity_predictor with replay buffer
        current_iter_loss = []
        for _ in range(self._optim_proximity_epochs * num_batches):
            sampled_fail_states = proximity.sample_fail_batch(optim_batchsize)
            sampled_success_states = proximity.sample_success_batch(optim_batchsize)
            fetched = loss(sampled_fail_states, sampled_success_states)
            current_iter_loss.append(fetched['fake_loss']+fetched['real_loss'])
            adam.update(fetched['g'],
                        self._optim_proximity_stepsize * self._cur_proximity_lrmult)
            if self._is_chef:
                for key, value in fetched.items():
                    if key != 'g':
                        if np.isscalar(value):
                            info[key].append(value)
                        else:
                            info[key].extend(value)
                    else:
                        grad_norm_value = np.linalg.norm(value)
                        info['grad_norm'].append(grad_norm_value)
                        info['grad_norm_clipped'].append(np.clip(
                            grad_norm_value, 0, self._config.proximity_max_grad_norm))
        proximity.last_iter_loss = np.average(current_iter_loss)

        if self._is_chef:
            logger.warn('proximity.last_iter_loss: {}'.format(proximity.last_iter_loss))
            info['batch_size'] = [optim_batchsize]
            info['buffer_size_success_final'] = [proximity.success_buffer.size()]
            info['buffer_size_fail_final'] = [proximity.fail_buffer.size()]
            # hist summary
            if len(cur_primitive == idx) > 0:
                if rew_success_final.shape[0] > 0:
                    info['hist_success_final'] = rew_success_final
                if rew_success_intermediate.shape[0] > 0:
                    info['hist_success_intermediate'] = rew_success_intermediate
                if rew_fail_final.shape[0] > 0:
                    info['hist_fail_final'] = rew_fail_final
                if rew_fail_intermediate.shape[0] > 0:
                    info['hist_fail_intermediate'] = rew_fail_intermediate
            return info
        return None

    def _update_proximity_predictor_network(self, segs, num_batches,
                                            only_use_trans_term_state):
        assert self._use_trans
        logger.info("===== Optimizing proximity_predictors ... =====")

        if self._is_chef:
            info = defaultdict(list)

        for i in range(len(self.proximity_predictors)):
            _info = self._update_one_proximity_predictor_network(
                segs[i], self.proximity_predictors[i],
                self._proximity_predictor_adams[i],
                self._proximity_predictor_losses[i],
                self._proximity_predictor_zerograds[i],
                num_batches, i,
                only_use_trans_term_state=only_use_trans_term_state)

            if _info is not None:
                for key, value in _info.items():
                    info['proximity_predictor_{}/'.format(
                        self.proximity_predictors[i].env_name,) + key].extend(value)

        logger.info("===== Optimizing proximity_predictors done =====")

        if self._is_chef:
            for key, value in info.items():
                if 'hist' not in key:
                    info[key] = np.mean(value)
            info['proximity_predictor_{}/global_norm'.format(
                self.proximity_predictors[i].env_name)] = self._proximity_predictor_global_norm()
            return info
        return None

    def _update_one_trans_policy(self, seg, pi, adam, loss, zero_grad, update_trans_oldpi, num_batches):
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        cur_primitive, term = seg["cur_primitive"], seg["term"]
        if self._is_chef:
            info = defaultdict(list)

        optim_batchsize = min(self._optim_batchsize, ob.shape[0])
        logger.log("Optimizing trans_{}... {} epochs * {} batches * {} batchsize <- {} data".format(
            pi.env_name, self._optim_epochs, num_batches, optim_batchsize, ob.shape[0]))

        if np.shape(ob)[0] == 0:
            logger.warn('[!] No transition is used')
            for ob_name in pi.ob_type:
                pi.ob_rms[ob_name].noupdate()
            blank = zero_grad()
            for _ in range(self._optim_epochs):
                for _ in range(num_batches):
                    adam.update(blank, self._optim_stepsize * self._cur_lrmult)
            return None

        # normalize advantage
        atarg = (atarg - atarg.mean()) / max(atarg.std(), 0.000001)

        # prepare batches
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret,
                         cur_primitive=cur_primitive, term=term), shuffle=True)

        ob_dict = self._env.get_ob_dict(ob)
        for ob_name in pi.ob_type:
            pi.ob_rms[ob_name].update(ob_dict[ob_name])

        update_trans_oldpi()
        b = 0
        for _ in range(self._optim_epochs):
            for batch in d.iterate_once(optim_batchsize):
                if b == self._optim_epochs * num_batches:
                    break
                ob_list = pi.get_ob_list(batch["ob"])
                fetched = loss(
                    self._cur_lrmult, batch["cur_primitive"], batch["ac"],
                    batch["atarg"], batch["vtarg"], batch["term"], *ob_list)
                adam.update(fetched['g'], self._optim_stepsize * self._cur_lrmult)
                b += 1
                if self._is_chef:
                    for key, value in fetched.items():
                        if key != 'g':
                            if np.isscalar(value):
                                info[key].append(value)
                            else:
                                info[key].extend(value)
                        else:
                            grad_norm_value = np.linalg.norm(value)
                            info['grad_norm'].append(grad_norm_value)
                            info['grad_norm_clipped'].append(np.clip(
                                grad_norm_value, 0, self._config.trans_max_grad_norm))
        blank = zero_grad()
        for _ in range(self._optim_epochs * num_batches - b):
            adam.update(blank, self._optim_stepsize * self._cur_lrmult)

        term_pred = seg["term"]
        batchsize = optim_batchsize

        if self._is_chef:
            info['term_pred'] = [np.mean(term_pred)]
            info['batch_size'] = [batchsize]
            return info
        return None

    def _update_trans_policies(self, segs, num_batches):
        assert self._use_trans
        logger.info("Optimizing trans...")

        if self._is_chef:
            info = defaultdict(list)

        for i in range(len(self.trans_pis)):
            _info = self._update_one_trans_policy(
                segs[i], self.trans_pis[i], self.trans_adams[i],
                self.trans_losses[i], self.trans_zerograds[i],
                self._update_old_trans[i], num_batches)

            if _info is not None:
                for key, value in _info.items():
                    info['trans_{}/'.format(
                        self.trans_pis[i].env_name) + key].extend(value)
                info['trans_{}/global_norm'.format(
                    self.trans_pis[i].env_name)] = self._trans_global_norm()

        if self._is_chef:
            for key, value in info.items():
                info[key] = np.mean(value)
            return info
        return None


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
