#!/bin/bash
python run.py --mpi 4 --prefix pick_ICLR2019 --env JacoPick-v1 --num_rollouts 10000 --hrl False
python run.py --mpi 4 --prefix catch_ICLR2019 --env JacoCatch-v1 --num_rollouts 10000 --hrl False
python run.py --mpi 4 --prefix toss_ICLR2019 --env JacoToss-v1 --num_rollouts 10000 --hrl False
python run.py --mpi 4 --prefix hit_ICLR2019 --env JacoHit-v1 --num_rollouts 10000 --hrl False
