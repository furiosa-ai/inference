#!/bin/bash

# define env. variables
git_dir=$(git rev-parse --show-toplevel)
work_dir=$git_dir/language/gpt-j
env_name=mlperf-gptj
conda_base=$($CONDA_EXE info --base)

# work on gpt-j
cd $work_dir

# create and enter conda env.
printf "\n============= STEP-1: Create conda environment and activate =============\n"
conda remove -n mlperf-gptj --all -y
rm -rf $conda_base/env/$env_name
conda env create -f eval_environment.yml
set +u
source "$conda_base/etc/profile.d/conda.sh"
conda activate $env_name
set -u

# build mlperf loadgen
printf "\n============= STEP-2: Build mlperf loadgen =============\n"
pip install pybind11==2.11.1
cd $git_dir/loadgen; python setup.py install
cd -

# pull model and dataset
printf "\n============= STEP-3: Pull dvc data =============\n"
pip install dvc[s3]
dvc pull model --force
dvc pull data --force

# eval gpt-j
printf "\n============= STEP-4: Run eval =============\n"
DATASET_PATH=./data/cnn_eval.json
SCENARIO=Offline
LOG_PATH=build/logs/$SCENARIO python main.py --scenario=$SCENARIO --model-path=./model --dataset-path=$DATASET_PATH --accuracy --gpu
python evaluation.py --mlperf-accuracy-file=$LOG_PATH/mlperf_log_accuracy.json --dataset-file=$DATASET_PATH

printf "\n=============End of eval =============\n"

# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir
