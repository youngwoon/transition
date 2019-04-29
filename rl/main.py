import sys
import os
import os.path as osp
import time
import pipes
from six.moves import shlex_quote

import cv2  # Should be here for rollouts.py
from mpi4py import MPI
import tensorflow as tf
import h5py

import baselines.common.tf_util as U
from baselines.common import set_global_seeds
from baselines import logger
from baselines.common.atari_wrappers import TransitionEnvWrapper

from rl.meta_policy import MetaPolicy
from rl.primitive_policy import PrimitivePolicy
from rl.mlp_policy import MlpPolicy
from rl.transition_policy import TransitionPolicy
from rl.proximity_predictor import ProximityPredictor
from rl.config import argparser
from rl.util import make_env
import rl.rollouts as rollouts


def load_buffers(proximity_predictors, ckpt_path):
    if proximity_predictors:
        buffer_path = ckpt_path + '.hdf5'
        if os.path.exists(buffer_path):
            logger.info('Load buffers from {}'.format(buffer_path))
            with h5py.File(buffer_path, 'r') as buffer_file:
                for p in proximity_predictors:
                    success_obs = buffer_file[p.env_name]['success'].value
                    fail_obs = buffer_file[p.env_name]['fail'].value
                    if success_obs.shape[0]:
                        p.success_buffer.add(success_obs)
                    if fail_obs.shape[0]:
                        p.fail_buffer.add(fail_obs)
                    logger.info('Load buffers for {}. success states ({})  fail states ({})'.format(
                        p.env_name, success_obs.shape[0], fail_obs.shape[0]))
        else:
            logger.warn('No buffers are available at {}'.format(buffer_path))


def run(config):
    sess = U.single_threaded_session(gpu=False)
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    is_chef = (rank == 0)

    workerseed = config.seed + 10000 * rank
    set_global_seeds(workerseed)

    if is_chef:
        logger.configure()
    else:
        logger.set_level(logger.DISABLED)
        config.render = False
        config.record = False

    env_name = config.env
    env = make_env(env_name, config)

    if is_chef and config.is_train:
        with open(osp.join(config.log_dir, "args.txt"), "a") as f:
            f.write("\nEnvironment argument:\n")
            for k in sorted(env.unwrapped._config.keys()):
                f.write("{}: {}\n".format(k, env.unwrapped._config[k]))

    networks = []

    # build models
    if config.hrl:
        assert config.primitive_envs is not None and config.primitive_paths is not None

        logger.info('====== Module list ======')
        num_primitives = len(config.primitive_envs)
        for primitive_env_name, primitive_path in zip(config.primitive_envs, config.primitive_paths):
            logger.info('Env: {}, Dir: {}'.format(primitive_env_name, primitive_path))

        meta_pi = MetaPolicy(
            name="%s/meta_pi" % env_name,
            env=env,
            ob_env_name=env_name,
            primitives=config.primitive_envs,
            config=config)

        meta_oldpi = MetaPolicy(
            name="%s/meta_oldpi" % env_name,
            env=env,
            ob_env_name=env_name,
            primitives=config.primitive_envs,
            config=config)

        primitive_pis = [
            PrimitivePolicy(
                name="%s/pi" % primitive_env_name,
                env=env,
                ob_env_name=primitive_env_name,
                config=config)
            for primitive_env_name in config.primitive_envs]

        trans_pis, trans_oldpis = None, None
        if config.use_trans:
            trans_pis = [
                TransitionPolicy(
                    name="%s/transition_pi" % primitive_env_name,
                    env=env,
                    ob_env_name=env_name if config.trans_include_task_obs else primitive_env_name,
                    num_primitives=num_primitives,
                    trans_term_activation=config.trans_term_activation,
                    config=config)
                for primitive_env_name in config.primitive_envs]
            trans_oldpis = [
                TransitionPolicy(
                    name="%s/transition_oldpi" % primitive_env_name,
                    env=env,
                    ob_env_name=env_name if config.trans_include_task_obs else primitive_env_name,
                    num_primitives=num_primitives,
                    trans_term_activation=config.trans_term_activation,
                    config=config)
                for primitive_env_name in config.primitive_envs]
            networks.extend(trans_pis)
            networks.extend(trans_oldpis)
        networks.append(meta_pi)
        networks.append(meta_oldpi)
        networks.extend(primitive_pis)

        # build proximity_predictor
        proximity_predictors = None
        if config.use_proximity_predictor:
            portion_start = [float(v) for v in config.proximity_use_traj_portion_start]
            portion_end = [float(v) for v in config.proximity_use_traj_portion_end]
            if len(portion_start) == 1:
                portion_start = portion_start * num_primitives
            if len(portion_end) == 1:
                portion_end = portion_end * num_primitives

            proximity_predictors = [
                ProximityPredictor(
                    name="%s/proximity_predictor" % primitive_env_name,
                    path=path,
                    env=env,
                    ob_env_name=primitive_env_name,  # make env for every primitive
                    use_traj_portion_end=portion_end,
                    use_traj_portion_start=portion_start,
                    is_train=config.is_train,
                    config=config
                ) for primitive_env_name, path, portion_start, portion_end in \
                zip(config.primitive_envs, config.primitive_paths, portion_start, portion_end)]
            networks.extend(proximity_predictors)

        # build trainer
        from rl.trainer import Trainer
        trainer = Trainer(env, meta_pi, meta_oldpi,
                          proximity_predictors, num_primitives,
                          trans_pis, trans_oldpis, config)

        # build rollout
        rollout = rollouts.traj_segment_generator(
            # stochastic=config.is_train, config=config)
            env, meta_pi, primitive_pis, trans_pis,
            stochastic=True, config=config,
            proximity_predictors=proximity_predictors,
        )
    else:
        # build vanilla TRPO
        policy = MlpPolicy(
            env=env,
            name="%s/pi" % env_name,
            ob_env_name=env_name,
            config=config)

        old_policy = MlpPolicy(
            env=env,
            name="%s/oldpi" % env_name,
            ob_env_name=env_name,
            config=config)
        networks.append(policy)
        networks.append(old_policy)

        # build trainer
        from rl.trainer_rl import RLTrainer
        trainer = RLTrainer(env, policy, old_policy, config)
        # build rollout
        rollout = rollouts.traj_segment_generator_rl(
            # env, policy, stochastic=config.is_train, config=config)
            env, policy, stochastic=not config.is_collect_state, config=config)

    # initialize models
    def load_model(load_model_path, var_list=None):
        if os.path.isdir(load_model_path):
            ckpt_path = tf.train.latest_checkpoint(load_model_path)
        else:
            ckpt_path = load_model_path
        if ckpt_path:
            U.load_state(ckpt_path, var_list)
        return ckpt_path

    if config.load_meta_path is not None:
        var_list = meta_pi.get_variables() + meta_oldpi.get_variables()
        ckpt_path = load_model(config.load_meta_path, var_list)
        logger.info('* Load the meta policy from checkpoint: {}'.format(ckpt_path))

    def tensor_description(var):
        description = '({} [{}])'.format(
            var.dtype.name, 'x'.join([str(size) for size in var.get_shape()]))
        return description

    var_list = []
    for network in networks:
        var_list += network.get_variables()
    if is_chef:
        for var in var_list:
            logger.info('{} {}'.format(var.name, tensor_description(var)))

    if config.load_model_path is not None:
        # Load all the network
        if config.is_train:
            ckpt_path = load_model(config.load_model_path)
            if config.hrl:
                load_buffers(proximity_predictors, ckpt_path)
        else:
            ckpt_path = load_model(config.load_model_path, var_list)
        logger.info('* Load all policies from checkpoint: {}'.format(ckpt_path))
    elif config.is_train:
        ckpt_path = tf.train.latest_checkpoint(config.log_dir)
        if config.hrl:
            if ckpt_path:
                ckpt_path = load_model(ckpt_path)
                load_buffers(proximity_predictors, ckpt_path)
            else:
                # Only load the primitives
                for (primitive_name, primitive_pi) in zip(config.primitive_paths, primitive_pis):
                    var_list = primitive_pi.get_variables()
                    if var_list:
                        primitive_path = osp.expanduser(osp.join(config.primitive_dir, primitive_name))
                        ckpt_path = load_model(primitive_path, var_list)
                        logger.info("* Load module ({}) from {}".format(primitive_name, ckpt_path))
                    else:
                        logger.info("* Hard-coded module ({})".format(primitive_name))
            logger.info("Loading modules is done.")
        else:
            if ckpt_path:
                ckpt_path = load_model(ckpt_path)
    else:
        logger.info('[!] Checkpoint for evaluation is not provided.')
        ckpt_path = load_model(config.log_dir, var_list)
        logger.info("* Load all policies from checkpoint: {}".format(ckpt_path))

    if config.is_train:
        trainer.train(rollout)
    else:
        if config.evaluate_proximity_predictor:
            trainer.evaluate_proximity_predictor(var_list)
        else:
            trainer.evaluate(rollout, ckpt_num=ckpt_path.split('/')[-1])

    env.close()


def encode_args(args_str):
    args_dict = {}
    args_list = args_str.split("/")
    for args in args_list:
        k, v = args.split('-')
        args_dict.update({k: float(v)})
    return args_dict


def main():
    args = argparser()

    logger.info('Launch process {}/{}'.format(MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()))

    # set checkpoint dir and log dir
    env_name = args.env.split('-')[0]
    if args.use_proximity_predictor is True:
        env_name += '.proximity'

    if args.env_args is not None:
        env_name = '{}.{}.{}'.format(env_name,
                                     args.prefix or "",
                                     args.env_args.replace('/', '_'))
        args.env_args = encode_args(args.env_args)
    else:
        if args.prefix is not None:
            env_name = '{}.{}'.format(env_name, args.prefix)

    args.log_dir = osp.join(args.log_dir, env_name)

    if MPI.COMM_WORLD.Get_rank() == 0:
        # save the commands
        if args.is_train:
            os.makedirs(args.log_dir, exist_ok=True)
            train_cmd = 'python3 -m rl.main ' + ' '.join([pipes.quote(s) for s in sys.argv[1:]])

            def change_arg(cmd, arg_name, init_val, target_val):
                if arg_name in cmd:
                    cmd = cmd.replace('{} {}'.format(arg_name, init_val), '{} {}'.format(arg_name, target_val))
                    cmd = cmd.replace('{}={}'.format(arg_name, init_val), '{} {}'.format(arg_name, target_val))
                else:
                    cmd = '{} --{} {}'.format(cmd, arg_name, target_val)
                return cmd

            test_cmd = change_arg(train_cmd, 'is_train', 'True', 'False')
            test_cmd = change_arg(test_cmd, 'record', 'False', 'True')

            train_cmd += '\n'
            test_cmd += '\n'

            logger.info('\n' + '*' * 80)
            logger.info('Training command:\n' + train_cmd)
            logger.info('Testing command:\n' + test_cmd)
            logger.info('*' * 80 + '\n')

            with open(osp.join(args.log_dir, "cmd.txt"), "a+") as f:
                f.write(train_cmd)
            with open(osp.join(args.log_dir, "eval_cmd.txt"), "a+") as f:
                f.write(test_cmd)

            args_lines = ""
            args_lines += "Date and Time:\n"
            args_lines += time.strftime("%d/%m/%Y\n")
            args_lines += time.strftime("%H:%M:%S\n\n")
            for k, v in args.__dict__.items():
                if k != "env_args":
                    args_lines += "{}: {}\n".format(k, v)

            with open(osp.join(args.log_dir, "args.txt"), "w") as f:
                f.write(args_lines)

            # save code revision
            logger.info('Save git commit and diff to {}/git.txt'.format(args.log_dir))
            cmds = ["echo `git rev-parse HEAD` >> {}".format(
                        shlex_quote(osp.join(args.log_dir, 'git.txt'))),
                    "git diff >> {}".format(
                        shlex_quote(osp.join(args.log_dir, 'git.txt')))]
            os.system("\n".join(cmds))

    run(args)


if __name__ == '__main__':
    main()
