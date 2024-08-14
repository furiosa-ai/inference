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


quant_data_dvc_dir=quantized/BERT-large/mlperf_submission/W8A8
printf "\n============= Download quant_config from furiosa-llm-models artifacts=============\n"
#Pull quant config files from dvc
cd $git_dir
git clone https://github.com/furiosa-ai/furiosa-llm-models-artifacts.git
cd $git_dir/furiosa-llm-models-artifacts

git checkout $tag
dvc pull $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/quant_config.yaml.dvc -r origin --force
dvc pull $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/24L/qformat.yaml.dvc -r origin --force
dvc pull $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/24L/qparam.npy.dvc -r origin --force


mkdir -p $quant_data_dir
cp $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/quant_config.yaml $quant_data_dir/quant_config.yaml
cp $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/24L/qformat.yaml $quant_data_dir/calibration_range/quant_format.yaml
cp $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/24L/qparam.npy $quant_data_dir/calibration_range/quant_param.npy
rm -rf $git_dir/furiosa-llm-models-artifacts




# eval model
printf "\n============= STEP-4: Run eval =============\n"
SCENARIO=${SCENARIO:=Offline}
BACKEND="rngd"
MODEL_PATH=$data_dir/models/bert/model.pytorch
MODEL_CONFIG_PATH=$data_dir/models/bert/bert_config.json
VOCAB_PATH=$data_dir/models/bert/vocab.txt
DATASET_PATH=$data_dir/dataset/squad/validation/dev-v1.1.json
LOG_PATH=$log_dir/$model_name/$SCENARIO/$(date +%Y%m%d_%H%M%S%Z)
N_COUNT=${N_COUNT:="10833"} # total_len = 10,833
N_COUNT=100

# quantization args
CALIBRATE=${CALIBRATE:=false}
N_CALIB=${N_CALIB:=100} # total_len = 100
CALIB_DATA_PATH=$data_dir/dataset/squad/calibration/cal_features.pickle
QUANT_CONFIG_PATH=$quant_data_dir/quant_config.yaml
QUANT_PARAM_PATH=$quant_data_dir/calibration_range/quant_param.npy
QUANT_FORMAT_PATH=$quant_data_dir/calibration_range/quant_format.yaml

printf "<<EVAL_CONFIG>>\n"
printf "\tSCENARIO: $SCENARIO\n"
printf "\tNUM_EVAL_DATA: $N_COUNT\n"
printf "\tCALIBRATE: $CALIBRATE\n"

export LOG_PATH=$LOG_PATH
export ML_MODEL_FILE_WITH_PATH=$MODEL_PATH
export VOCAB_FILE=$VOCAB_PATH
export DATASET_FILE=$DATASET_PATH
export SKIP_VERIFY_ACCURACY=true

mkdir -p $LOG_PATH/calibration_range

if [ "$CALIBRATE" = true ]; then
    printf "\t\tNUM_CALIB_DATA: $N_CALIB\n"
    QUANT_PARAM_PATH=$LOG_PATH/calibration_range/quant_param.npy
    QUANT_FORMAT_PATH=$LOG_PATH/calibration_range/quant_format.yaml
    python -m quantization.calibrate --backend=$BACKEND \
                                     --model_path=$MODEL_PATH \
                                     --model_config_path=$MODEL_CONFIG_PATH \
                                     --quant_config_path=$QUANT_CONFIG_PATH \
                                     --quant_param_path=$QUANT_PARAM_PATH \
                                     --quant_format_path=$QUANT_FORMAT_PATH \
                                     --calib_data_path=$CALIB_DATA_PATH \
                                     --n_calib=$N_CALIB \
                                     --gpu
    printf "Save calibration range to $LOG_PATH/calibration_range"
else
    cp $QUANT_PARAM_PATH $LOG_PATH/calibration_range/quant_param.npy
    cp $QUANT_FORMAT_PATH $LOG_PATH/calibration_range/quant_format.yaml
fi

SECONDS=0
# N_COUNT=10
python -m run --scenario=$SCENARIO \
              --backend=$BACKEND \
              --gpu \
              --quantize \
              --quant_param_path=$QUANT_PARAM_PATH \
              --quant_format_path=$QUANT_FORMAT_PATH \
              --max_examples=$N_COUNT \
              --accuracy
duration=$SECONDS
printf "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed." &> $LOG_PATH/elapsed_time.log

ACCURACY_LOG_FILE=$LOG_PATH/mlperf_log_accuracy.json
python accuracy-squad.py --vocab_file=$VOCAB_PATH \
                         --val_data=$DATASET_PATH \
                         --log_file=$ACCURACY_LOG_FILE \
                         --out_file=$LOG_PATH/predictions.json \
                         --max_examples=$N_COUNT \
                         &> $LOG_PATH/accuracy_result.log

printf "Save evaluation log to $LOG_PATH"

unset LOG_PATH
unset ML_MODEL_FILE_WITH_PATH
unset VOCAB_FILE
unset DATASET_FILE
unset SKIP_VERIFY_ACCURACY

printf "\n============= End of eval =============\n"

# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir
