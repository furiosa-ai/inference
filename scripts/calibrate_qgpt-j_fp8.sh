#!/bin/bash

# define env. variables
model_name=qgpt-j
model_dir=language/gpt-j
git_dir=$(git rev-parse --show-toplevel)
work_dir=$git_dir/$model_dir
data_dir=$git_dir/data
quant_data_dir=$data_dir/quantization/gpt-j
log_dir=$git_dir/logs
env_name=mlperf-$model_name
conda_base=$($CONDA_EXE info --base)
quant_data_folder=quantized/GPT-J/mlperf_submission_slice/W8fA8fKV8f
# work on model directory

# enter existing conda env.
source "$conda_base/etc/profile.d/conda.sh"
conda activate $env_name

printf "\n============= Download quant_config from furiosa-llm-models artifacts=============\n"
#Pull quant config files from dvc
cd $git_dir
git clone https://github.com/furiosa-ai/furiosa-llm-models-artifacts.git
cd $git_dir/furiosa-llm-models-artifacts
#Test coce
tag=34e0f53

git checkout $tag
dvc pull $git_dir/furiosa-llm-models-artifacts/$quant_data_folder/quant_config.yaml.dvc -r origin --force

mkdir -p $quant_data_dir
cp $git_dir/furiosa-llm-models-artifacts/$quant_data_folder/quant_config.yaml $quant_data_dir/quant_config.yaml
rm -rf $git_dir/furiosa-llm-models-artifacts

# eval model
printf "\n============= Run calibration qgpt-j =============\n"
SCENARIO=${SCENARIO:="Offline"}
MODEL_PATH=$data_dir/models/gpt-j
DATASET_PATH=$data_dir/dataset/cnn-daily-mail/validation/cnn_eval.json
LOG_PATH=$log_dir/$model_name/$SCENARIO/$(date +%Y%m%d_%H%M%S%Z)
cd $work_dir
# quantization args
N_CALIB=${N_CALIB:=1000} # total_len=1,000
N_CALIB=10
CALIB_DATA_PATH=$data_dir/dataset/cnn-daily-mail/calibration/cnn_dailymail_calibration.json
QUANT_CONFIG_PATH=$quant_data_dir/quant_config.yaml
QUANT_PARAM_PATH=$quant_data_dir/calibration_range/quant_param.npy
QUANT_FORMAT_PATH=$quant_data_dir/calibration_range/quant_format.yaml

printf "<<CALIB_CONFIG>>\n"
printf "\tSCENARIO: $SCENARIO\n"
printf "\tCALIBRATE: $CALIBRATE\n"

export LOG_PATH

mkdir -p $LOG_PATH/calibration_range

printf "\tNUM_CALIB_DATA: $N_CALIB\n"
QUANT_PARAM_PATH=$LOG_PATH/calibration_range/quant_param.npy
QUANT_FORMAT_PATH=$LOG_PATH/calibration_range/quant_format.yaml
python -m quantization.calibrate --model_path=$MODEL_PATH \
                                    --quant_config_path=$QUANT_CONFIG_PATH \
                                    --quant_param_path=$QUANT_PARAM_PATH \
                                    --quant_format_path=$QUANT_FORMAT_PATH \
                                    --calib_data_path=$CALIB_DATA_PATH \
                                    --n_calib=$N_CALIB \
                                    --gpu \
                                    --save_cache_files
printf "Save calibration range to $LOG_PATH/calibration_range"

                                            


unset LOG_PATH
unset CALIBRATE

printf "\n============= End of calibration =============\n"

# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir
