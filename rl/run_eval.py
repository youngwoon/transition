from glob import glob
import os
import shutil
from six.moves import shlex_quote


# NEED TO BE CONFIGURED
LOG_DIR = 'log'
FINAL = True
NO_TRANS = False
START_IDX = 0
END_IDX = 1000

files = glob(os.path.join(LOG_DIR, '*/eval_cmd.txt'))
files.sort()

tmux_cmds = []
exp_cmds = []
tmux_cmds.append("tmux new-session -s {} -n {} -d {}".format('exp', 'dummy', 'bash'))
for idx, name in enumerate(files):
    if NO_TRANS and not ('ours' in name or 'OURS' in name):
        continue
    if START_IDX <= idx and idx < END_IDX:
        with open(name, 'r') as f:
            cmd = f.readlines()[-1].strip()
            if NO_TRANS:
                cmd += ' --test_module_net True'
            if FINAL:
                cmd += ' --num_evaluation_run 50 --final_eval True'
            else:
                cmd = cmd + ' --record True --num_evaluation_run 5'
            tmux_cmds.append("tmux new-window -t {}:{} -n {}".format('exp', idx, 'bash'))
            exp_cmds.append("tmux send-keys -t {}:{} {} Enter".format(
                'exp', idx, shlex_quote(cmd)))
            print(cmd)

tmux_cmds += ["sleep 1"]
print("\n".join(tmux_cmds))
print("\n".join(exp_cmds))
input("[!] Continue?")

os.system("\n".join(tmux_cmds))
os.system("\n".join(exp_cmds))
