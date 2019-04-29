#!/bin/bash -x
method=$1
v=$2
seeds=(123 456 789)

if [ $v = 1 ]
then
    env_name="Walker2dPatrol-v1"
    module_name="Walker2dForward-v1,Walker2dBackward-v1,Walker2dBalance-v1"
    module_path="Walker2dForward.forward_ICLR2019,Walker2dBackward.backward_ICLR2019,Walker2dBalance.balance_ICLR2019"
    traj_portion="0.2,0.2,1.0"
elif [ $v = 2 ]
then
    env_name="Walker2dHurdle-v1"
    module_name="Walker2dForward-v1,Walker2dJump-v1"
    module_path="Walker2dForward.forward_ICLR2019,Walker2dJump.jump_ICLR2019"
    traj_portion="0.2,0.4"
elif [ $v = 3 ]
then
    env_name="Walker2dObstacleCourse-v1"
    module_name="Walker2dForward-v1,Walker2dJump-v1,Walker2dCrawl-v1"
    module_path="Walker2dForward.forward_ICLR2019,Walker2dJump.jump_ICLR2019,Walker2dCrawl.crawl_ICLR2019"
    traj_portion="0.2,0.4,0.1"
fi

for seed in "${seeds[@]}"; do
    echo ${seed}
    if [ $method = trpo ]
    then
        python run.py --mpi 4 --prefix trpo_${env_name}_seed_${seed} --env ${env_name} --hrl False --seed ${seed} --max_iters 20001 --rl_method trpo --env_args sparse_reward-0
    elif [ $method = ppo ]
    then
        python run.py --mpi 4 --prefix ppo_${env_name}_seed_${seed} --env ${env_name} --hrl False --seed ${seed} --max_iters 20001 --rl_method ppo --optim_stepsize 3e-4 --env_args sparse_reward-0
    elif [ $method = task ]
    then
        python run.py --mpi 4 --prefix task_${env_name}_seed_${seed} --env ${env_name} --primitive_envs ${module_name} --primitive_paths ${module_path} --seed ${seed} --max_iters 10001 --lr_decay False --use_proximity_predictor False
    elif [ $method = sparse ]
    then
        python run.py --mpi 4 --prefix sparse_${env_name}_seed_${seed} --env ${env_name} --primitive_envs ${module_name} --primitive_paths ${module_path} --proximity_use_traj_portion_end ${traj_portion} --seed ${seed} --max_iters 10001 --proximity_dense_diff_rew False --lr_decay False
    elif [ $method = ours ]
    then
        python run.py --mpi 4 --prefix ours_${env_name}_seed_${seed} --env ${env_name} --primitive_envs ${module_name} --primitive_paths ${module_path} --proximity_use_traj_portion_end ${traj_portion} --lr_decay False --seed ${seed} --max_iters 3001
    fi
done
