#!/usr/bin/env bash
export OMP_NUM_THREADS=5
export PYTHONPATH=$(pwd)
model_dir=$1
shift
extra_args="$@"
net_run train -c config/train/config.ini -a applications.regression_motion_sim.Regress --model_dir=${model_dir} ${extra_args}