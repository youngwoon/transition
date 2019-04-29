#!/usr/bin/env python
import argparse
import os
import sys
from six.moves import shlex_quote
from datetime import datetime, timedelta


parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('--prefix', type=str, default='temp')
parser.add_argument('-n', '--dry-run', action='store_true',
                    help="Print out commands rather than executing them")
parser.add_argument('--mpi', type=int, default=1, help="Use mpi")
parser.add_argument('--gpu_id', type=int, default=-1, help="GPU id")
parser.add_argument('--display', type=str, default='1')


def get_time(gap=0):
    return (datetime.now() + timedelta(seconds=gap)).strftime("%Y-%m-%d_%H-%M-%S")


def new_cmd(session, name, cmd, shell):
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(shlex_quote(str(v)) for v in cmd)
    return name, "tmux send-keys -t {}:{} {} Enter".format(session, name, shlex_quote(cmd))


def create_commands(session, args, unparsed, shell='bash', mode="tmux"):
    # for launching the TF workers and for launching tensorboard
    base_cmd = [sys.executable, '-m', 'rl.main', '--prefix', session] + unparsed

    session = '{}_{}'.format(session, get_time())

    cmds_map = []
    train_cmd = base_cmd
    if args.mpi > 1:
        train_cmd = ['mpirun', '-np', args.mpi] + base_cmd
    train_cmd = ['CUDA_VISIBLE_DEVICES={}'.format(args.gpu_id)] + train_cmd

    cmds_map += [new_cmd(session, "train", train_cmd, shell)]
    cmds_map += [new_cmd(session, "eval", ['#'] + base_cmd + ['--is_train', 'False',
                                                              '--record', 'True'], shell)]

    windows = [v[0] for v in cmds_map]

    notes = []
    notes += ["Use `tmux attach -t {}` to watch process output".format(session)]
    notes += ["Use `tmux kill-session -t {}` to kill the job".format(session)]

    cmds = [
        "tmux kill-session -t {}".format(session),
        "tmux new-session -s {} -n {} -d {}".format(session, windows[0], shell)
    ]
    for w in windows[1:]:
        cmds += ["tmux new-window -t {} -n {} {}".format(session, w, shell)]
    cmds += ["sleep 1"]

    for window, cmd in cmds_map:
        cmds += [cmd]

    return cmds, notes


def main():
    args, unparsed = parser.parse_known_args()
    cmds, notes = create_commands(args.prefix, args, unparsed)
    if args.dry_run:
        print("Dry-run mode due to -n flag, otherwise the following commands would be executed:")
    else:
        print("Executing the following commands:")
    print("\n".join(cmds))
    print("")
    if not args.dry_run:
        os.environ["TMUX"] = ""
        os.environ["DISPLAY"] = ":{}".format(args.display)
        os.system("\n".join(cmds))
    print('\n'.join(notes))


if __name__ == '__main__':
    main()
