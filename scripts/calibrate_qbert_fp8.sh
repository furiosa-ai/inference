#!/bin/bash

# define env. variables
model_name=qbert
model_dir=language/bert
git_dir=$(git rev-parse --show-toplevel)
work_dir=$git_dir/$model_dir
data_dir=$git_dir/data
quant_data_dir=$data_dir/quantization/bert
log_dir=$git_dir/logs
env_name=mlperf-$model_name
conda_base=$($CONDA_EXE info --base)
tag=MLPerf4.1-v4.2
quant_data_dvc_dir=quantized/BERT-large/mlperf_submission/W8fA8f

# work on model directory
cd $work_dir

# enter existing conda env.
source "$conda_base/etc/profile.d/conda.sh"
conda activate $env_name

printf "\n============= Download quant_config from furiosa-llm-models artifacts=============\n"
#Pull quant config files from dvc
cd $git_dir
git clone https://github.com/furiosa-ai/furiosa-llm-models-artifacts.git
cd $git_dir/furiosa-llm-models-artifacts

git checkout $tag
dvc pull $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/quant_config.yaml.dvc -r origin --force
mkdir -p $quant_data_dir
cp $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/quant_config.yaml $quant_data_dir/quant_config.yaml
rm -rf $git_dir/furiosa-llm-models-artifacts

# eval model
printf "\n============= Run calibration qbert =============\n"
SCENARIO=${SCENARIO:=Offline}
MODEL_PATH=$data_dir/models/bert/model.pytorch
MODEL_CONFIG_PATH=$data_dir/models/bert/bert_config.json
LOG_PATH=$log_dir/$model_name/$SCENARIO/$(date +%Y%m%d_%H%M%S%Z)

# quantization args
N_CALIB=${N_CALIB:=100} # total_len = 100
N_CALIB=100
CALIB_DATA_PATH=$data_dir/dataset/squad/calibration/cal_features.pickle
QUANT_CONFIG_PATH=$quant_data_dir/quant_config.yaml
QUANT_PARAM_PATH=$quant_data_dir/calibration_range/quant_param.npy
QUANT_FORMAT_PATH=$quant_data_dir/calibration_range/quant_format.yaml

export LOG_PATH=$LOG_PATH

mkdir -p $LOG_PATH/calibration_range

MODEL_TYPE="mlperf-submission" # could be erased at submission
printf "\t\tMODEL_TYPE: $MODEL_TYPE\n"
printf "\t\tNUM_CALIB_DATA: $N_CALIB\n"
QUANT_PARAM_PATH=$LOG_PATH/calibration_range/quant_param.npy
QUANT_FORMAT_PATH=$LOG_PATH/calibration_range/quant_format.yaml
python -m quantization.calibrate --model_type=$MODEL_TYPE \
                                    --model_path=$MODEL_PATH \
                                    --model_config_path=$MODEL_CONFIG_PATH \
                                    --quant_config_path=$QUANT_CONFIG_PATH \
                                    --quant_param_path=$QUANT_PARAM_PATH \
                                    --quant_format_path=$QUANT_FORMAT_PATH \
                                    --calib_data_path=$CALIB_DATA_PATH \
                                    --n_calib=$N_CALIB \
                                    --gpu \
                                    --save_cache_files
printf "Save calibration range to $LOG_PATH/calibration_range"
cd $work_dir


unset LOG_PATH

printf "\n============= End of calibration =============\n"

# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir
