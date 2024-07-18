#!/bin/bash

# define env. variables
model_name=mlperf-gpt-j
model_dir=language/gpt-j
git_dir=$(git rev-parse --show-toplevel)
work_dir=$git_dir/$model_dir
data_dir=$git_dir/data
quant_data_dir=$data_dir/quantization/gpt-j
log_dir=$git_dir/logs
env_name=mlperf-llm
conda_base=$($CONDA_EXE info --base)

# enter existing conda env.
source "$conda_base/etc/profile.d/conda.sh"
conda activate $env_name

# eval model
printf "\n============= STEP-4: Run eval =============\n"
SCENARIO=${SCENARIO:="Offline"}
BACKEND="rngd-npu"
MODEL_PATH=$data_dir/models/gpt-j
DATASET_PATH=$data_dir/dataset/cnn-daily-mail/validation/cnn_eval.json
LOG_PATH=$log_dir/$model_name/$SCENARIO/$(date +%Y%m%d_%H%M%S%Z)
DEVICES=${DEVICES:="npu:0:0-3, npu:0:0-3"}
N_COUNT=${N_COUNT:="13368"} # total_len=13,368

printf "<<EVAL_CONFIG>>\n"
printf "\tSCENARIO: $SCENARIO\n"
printf "\tNUM_EVAL_DATA: $N_COUNT\n"
printf "\tLLM_ENGINE_ARTIFACTS_PATH: $LLM_ENGINE_ARTIFACTS_PATH\n"
printf "\tDEVICE: $DEVICES\n"

export NPU_ARCH=renegade
export LLM_ENGINE_ARTIFACTS_PATH=${LLM_ENGINE_ARTIFACTS_PATH:-"$data_dir/compile/gptj"}
export DISABLE_PROFILER=1
export LOG_PATH

mkdir -p $LOG_PATH/calibration_range

SECONDS=0
python $work_dir/main.py --scenario=$SCENARIO \
               --backend=$BACKEND \
               --model-path=$MODEL_PATH \
               --dataset-path=$DATASET_PATH \
               --max_examples=$N_COUNT \
               --device=$DEVICES \
               --accuracy
duration=$SECONDS
printf "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed." &> $LOG_PATH/elapsed_time.log

ACCURACY_LOG_FILE=$LOG_PATH/mlperf_log_accuracy.json
python $work_dir/evaluation.py --mlperf-accuracy-file=$ACCURACY_LOG_FILE \
                     --dataset-file=$DATASET_PATH \
                     &> $LOG_PATH/accuracy_result.log

printf "Save eval log to $LOG_PATH"

unset LOG_PATH

printf "\n============= End of eval =============\n"

# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir
