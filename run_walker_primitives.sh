#!/bin/bash
python run.py --mpi 4 --prefix forward_ICLR2019 --env Walker2dForward-v1 --num_rollouts 10000 --hrl False
python run.py --mpi 4 --prefix backward_ICLR2019 --env Walker2dBackward-v1 --num_rollouts 10000 --hrl False
python run.py --mpi 4 --prefix jump_ICLR2019 --env Walker2dJump-v1 --num_rollouts 10000 --hrl False
python run.py --mpi 4 --prefix balance_ICLR2019 --env Walker2dBalance-v1 --num_rollouts 10000 --hrl False
python run.py --mpi 4 --prefix crawl_ICLR2019 --env Walker2dCrawl-v1 --num_rollouts 10000 --hrl False
