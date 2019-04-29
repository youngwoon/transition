import argparse
import os


def argparser():
    def str2bool(v):
        return v.lower() == 'true'

    def str2list(v):
        if not v:
            return v
        else:
            return [v_ for v_ in v.split(',')]

    parser = argparse.ArgumentParser("Modular Framework with Transition Policies",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # environment
    parser.add_argument('--env', help='environment ID', type=str, default='Walker2dForward-v1')
    parser.add_argument('--env_args', type=str, default=None, help='(optional) arguments for environment')

    # architecture (rl or hrl)
    parser.add_argument('--hrl', type=str2bool, default=True, help='Set to False to train a \
                        primitive policy or True to train transition policies for a complex skill')

    # vanilla mlp policy
    parser.add_argument('--rl_num_hid_layers', type=int, default=2)
    parser.add_argument('--rl_hid_size', type=int, default=32)
    parser.add_argument('--rl_activation', type=str, default='tanh',
                        choices=['relu', 'elu', 'tanh'])
    parser.add_argument('--rl_method', type=str, default='trpo',
                        choices=['ppo', 'trpo'])
    parser.add_argument('--rl_fixed_var', type=str2bool, default=True)

    # meta policy
    parser.add_argument('--meta_duration', type=int, default=30)
    parser.add_argument('--meta_oracle', type=str2bool, default=True)
    parser.add_argument('--meta_num_hid_layers', type=int, default=2)
    parser.add_argument('--meta_hid_size', type=int, default=32)
    parser.add_argument('--meta_activation', type=str, default='tanh',
                        choices=['relu', 'elu', 'tanh'])
    parser.add_argument('--meta_max_grad_norm', type=float, default=10.0)
    parser.add_argument('--meta_method', type=str, default='ppo',
                        choices=['ppo', 'trpo'])
    parser.add_argument('--meta_entcoeff', type=float, default=2e-4)

    # transition policy
    parser.add_argument('--use_trans', type=str2bool, default=True)
    parser.add_argument('--use_trans_between_same_policy', type=str2bool, default=False)
    parser.add_argument('--trans_term_activation', type=str, default='softmax',
                        choices=['sigmoid', 'softmax'])
    parser.add_argument('--trans_term_prob', type=float, default=0.02)
    parser.add_argument('--trans_duration', type=int, default=100)
    parser.add_argument('--trans_num_hid_layers', type=int, default=2)
    parser.add_argument('--trans_hid_size', type=int, default=32)
    parser.add_argument('--trans_activation', type=str, default='tanh',
                        choices=['relu', 'elu', 'tanh'])
    parser.add_argument('--trans_max_grad_norm', type=float, default=10.0)
    parser.add_argument('--trans_method', type=str, default='ppo',
                        choices=['ppo', 'trpo'])
    parser.add_argument('--trans_fixed_var', type=str2bool, default=True)
    parser.add_argument('--trans_entcoeff', type=float, default=1e-3)
    parser.add_argument('--trans_include_acc', type=str2bool, default=True)
    parser.add_argument('--trans_include_task_obs', type=str2bool, default=False)
    parser.add_argument('--trans_apply_first_time_step', type=str2bool, default=False)

    # proximity predictor
    parser.add_argument('--use_proximity_predictor', type=str2bool, default=True)
    parser.add_argument('--proximity_loss_type', type=str, default='lsgan',
                        choices=['vanilla', 'lsgan', 'wgan'])
    parser.add_argument('--proximity_use_traj_portion_start', type=str2list, default='0',
                        help='Fraction representing portion of collected trajectory rollout \
                              the proximity predictor trains on, specified for each primitive')
    parser.add_argument('--proximity_use_traj_portion_end', type=str2list, default='1',
                        help='Ending index capturing a fraction of the collected trajectory \
                              rollout that the proximity predictor trains on, specified for each primitive')
    parser.add_argument('--proximity_num_hid_layers', type=int, default=2)
    parser.add_argument('--proximity_hid_size', type=int, default=96)
    parser.add_argument('--proximity_activation_fn', type=str, default='relu',
                        choices=['relu', 'elu', 'leaky', 'tanh', 'sigmoid'])
    parser.add_argument('--proximity_learning_rate', type=float, default=1e-4)
    parser.add_argument('--proximity_optim_epochs', type=int, default=5)
    parser.add_argument('--proximity_hist', type=str2bool, default=False)
    parser.add_argument('--proximity_hist_num_bin', type=int, default=10)
    parser.add_argument('--proximity_dense_diff_rew', type=str2bool, default=True)
    parser.add_argument('--proximity_dense_diff_rew_final_bonus', type=str2bool, default=True)
    parser.add_argument('--proximity_include_acc', type=str2bool, default=True)
    parser.add_argument('--proximity_obs_norm', type=str2bool, default=True)
    parser.add_argument('--proximity_only_use_trans_term_state', type=str2bool, default=False)
    parser.add_argument('--proximity_replay_size', type=int, default=1000000)
    parser.add_argument('--proximity_keep_collected_obs', type=str2bool, default=True)
    parser.add_argument('--proximity_weight_decay_linear', type=str2bool, default=False)
    parser.add_argument('--proximity_weight_decay_rate', type=float, default=0.95)
    parser.add_argument('--proximity_max_grad_norm', type=float, default=10.0)

    # primitive skills
    parser.add_argument('--primitive_envs', type=str2list, default=None, help='Separated list \
                        of primitive envs eg. JacoToss-v1,JacoHit-v1')
    parser.add_argument('--primitive_dir', type=str, default='./log',
                        help='Directory where primitives are located')
    parser.add_argument('--primitive_paths', type=str2list, default=None, help='Separated list \
                        of model names inside primitive_dir loaded in order with primitive_envs \
                        eg. JacoToss.ICLR2019,JacoHit.ICLR2019')
    parser.add_argument('--primitive_num_hid_layers', type=int, default=2)
    parser.add_argument('--primitive_hid_size', type=int, default=32)
    parser.add_argument('--primitive_activation', type=str, default='tanh',
                        choices=['relu', 'elu', 'tanh'])
    parser.add_argument('--primitive_method', type=str, default='trpo',
                        choices=['ppo', 'trpo'])
    parser.add_argument('--primitive_fixed_var', type=str2bool, default=True)
    parser.add_argument('--primitive_include_acc', type=str2bool, default=False)
    parser.add_argument('--primitive_use_term', type=str2bool, default=True)

    # training
    parser.add_argument('--is_train', type=str2bool, default=True)
    parser.add_argument('--load_meta_path', type=str, default=None, help='Only load the meta controller')
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--write_summary_step', type=int, default=5)
    parser.add_argument('--ckpt_save_step', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10001)
    parser.add_argument('--num_rollouts', type=int, default=10000)
    parser.add_argument('--num_batches', type=int, default=32)
    parser.add_argument('--num_trans_batches', type=int, default=256)

    # evalution
    parser.add_argument('--num_evaluation_run', type=int, default=10)
    parser.add_argument('--evaluate_proximity_predictor', type=str2bool, default=False)
    parser.add_argument('--evaluate_all_ckpts', type=str2bool, default=False)
    parser.add_argument('--evaluation_log', type=str2bool, default=True)
    parser.add_argument('--video_caption_off', type=str2bool, default=False)
    parser.add_argument('--final_eval', type=str2bool, default=False)
    parser.add_argument('--test_module_net', type=str2bool, default=False)

    # collect states
    parser.add_argument('--is_collect_state', type=str2bool, default=False)

    # ppo
    parser.add_argument('--lr_decay', type=str2bool, default=True)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--entcoeff', type=float, default=0.0)
    parser.add_argument('--optim_epochs', type=int, default=10)
    parser.add_argument('--optim_stepsize', type=float, default=1e-4)
    parser.add_argument('--optim_batchsize', type=int, default=64)

    # trpo
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--cg_iters', type=int, default=10)
    parser.add_argument('--cg_damping', type=float, default=0.1)
    parser.add_argument('--vf_stepsize', type=float, default=1e-3)
    parser.add_argument('--vf_iters', type=int, default=5)

    # misc
    parser.add_argument('--prefix', type=str, default=None, help='Prefix for training files')
    parser.add_argument('--render', type=str2bool, default=False, help='Render frames')
    parser.add_argument('--record', type=str2bool, default=False, help='Record video')
    parser.add_argument('--video_prefix', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--debug', type=str2bool, default=False, help='See debugging info')

    args = parser.parse_args()
    args.env_args_str = args.env_args

    # save ckpt more frequently when training hrl
    if args.hrl:
        args.ckpt_save_step = 25
        if len(args.primitive_envs) == 1:
            args.use_trans_between_same_policy = True
    else:
        args.use_proximity_predictor = False

    if args.use_trans is False:
        args.use_proximity_predictor = False

    if args.is_collect_state:
        args.is_train = False
        args.record = False
        args.num_evaluation_run = 1000

    if args.final_eval:
        args.is_train = False
        args.record = False
        args.render = False
        args.num_evaluation_run = 50

    if args.render or args.record:
        import subprocess
        outputs = subprocess.run("ps ax | grep -Po '.*Xorg :\K(\d+|\d+.\d+)'", shell=True, stdout=subprocess.PIPE)
        displays = outputs.stdout.decode().split()
        if len(displays) != 1:
            print('Too many displays are available: {}'.format(', '.join(displays)))
        os.environ["DISPLAY"] = ":{}".format(displays[0])

    return args

