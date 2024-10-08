#!/bin/bash

#######################################################################
# LLaMA3.1-70B Forward Test Script
#
# This script performs the following steps:
# 1. Sets up environment variables and paths.
# 2. Activates the Conda environment.
# 3. Runs calibration if needed.
# 4. Checks the equivalence of outputs obtained at each generation step.
# 5. Cleans up and exits.
#
#######################################################################

set -e  # 에러 발생시 bash script 즉시 종료

##############################
# Step 0: DTYPE args 처리
##############################

# 기본 데이터 타입 설정
CONFIG_DTYPE="fp8"

# 인자가 제공되면 데이터 타입 설정
if [ $# -ge 1 ]; then
    CONFIG_DTYPE="$1"
    # 유효한 데이터 타입인지 확인
    if [ "$CONFIG_DTYPE" != "fp8" ] && [ "$CONFIG_DTYPE" != "int8" ]; then
        echo "유효하지 않은 dtype입니다: $CONFIG_DTYPE"
        echo "사용법: ./ci_bert.sh [dtype]"
        echo "  dtype: fp8 (기본값) 또는 int8"
        exit 1
    fi
fi

echo "사용할 데이터 타입: $CONFIG_DTYPE"

##############################
# Step 1: Define Environment Variables and Paths
##############################

# Define model and configuration parameters
MODEL_NAME="llama3.1-70b"
MODEL_DIR="language/llama2-70b"  #잘못입력한게 아닙니다. quant_config가 llama2, llama3.1 공통으로 공유가 되어서 그렇게됨.

# Define paths
GIT_ROOT_DIR=$(git rev-parse --show-toplevel)
WORK_DIR="$GIT_ROOT_DIR/$MODEL_DIR"
DATA_DIR="/home/home-mcl/shared_data/"
REF_PATH="$DATA_DIR/quant/llama3-70b/ref"
LOG_DIR="$GIT_ROOT_DIR/logs"
RES_PATH="$LOG_DIR/results"
QUANT_DATA_DIR="$DATA_DIR/quant/llama3-70b"

mkdir -p $RES_PATH

# Model and dataset paths
CHECKPOINT_PATH=/home/home-mcl/sunghyuck/inference/data/models/llama3/Meta-Llama-3.1-70B-Instruct #TMP 나중에 경로정리할거임
DATASET_PATH="$DATA_DIR/dataset/open-orca/validation/open_orca_gpt4_tokenized_llama.sampled_24576.pkl"
CALIB_DATA_PATH="$DATA_DIR/dataset/llama3-1/mgoin_ultrachat_2k_calibration_128.pkl"

# Scenario and device settings
SCENARIO="${SCENARIO:-Offline}"

# Evaluation settings
N_EVAL_DATA="${N_DATA:-1}"          # Number of evaluation data samples (default: 1)
N_CALIB_DATA="${N_CALIB:-128}"      # Number of calibration data samples (default: 128)
N_COUNT="${N_COUNT:-24576}"         # Total number of data samples (default: 24576)
SUBMISSION_MODEL_SOURCE="mlperf_submission_slice"

# Quantization configuration
QUANT_CONFIG_PATH="$QUANT_DATA_DIR/quant_config_${CONFIG_DTYPE}.yaml"

# Logging setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S%Z)
LOG_PATH="$LOG_DIR/$MODEL_NAME/$SCENARIO/W8fA8fKV8f/$TIMESTAMP"
CALIBRATION_RANGE_DIR="$LOG_PATH/calibration_range"
mkdir -p "$CALIBRATION_RANGE_DIR"

# Output directories
LOGIT_FOLDER_PATH="$WORK_DIR/ci_file/logit_files"
OUTPUT_FOLDER_PATH="$WORK_DIR/ci_file/output_files"
mkdir -p "$LOGIT_FOLDER_PATH"
mkdir -p "$OUTPUT_FOLDER_PATH"

# Move to the working directory
cd "$WORK_DIR"

##############################
# Step 2: Activate Conda Environment
##############################

echo -e "\n============= STEP 1: Activate Conda Environment ============="

export CONDA_EXE="/anaconda/condabin/conda"
CONDA_BASE=$($CONDA_EXE info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate inference-ci

##############################
# Step 3: Run Calibration (if needed)
##############################

echo -e "\n============= STEP 2: Run Calibration ============="

echo -e "<<EVAL_CONFIG>>"
echo -e "\tSCENARIO: $SCENARIO"
echo -e "\tNUM_EVAL_DATA: $N_EVAL_DATA"

export LOG_PATH

echo -e "\tNUM_CALIB_DATA: $N_CALIB_DATA"
# Update quantization parameter paths to save in the log directory
QUANT_PARAM_PATH="$CALIBRATION_RANGE_DIR/quant_param.npy"
QUANT_FORMAT_PATH="$CALIBRATION_RANGE_DIR/quant_format.yaml"
OUTPUT_PATH="$CALIBRATION_RANGE_DIR"

# Run calibration
python -m quantization.calibrate_llama3 \
    --model_path="$CHECKPOINT_PATH" \
    --quant_config_path="$QUANT_CONFIG_PATH" \
    --quant_param_path="$QUANT_PARAM_PATH" \
    --quant_format_path="$QUANT_FORMAT_PATH" \
    --calib_data_path="$CALIB_DATA_PATH" \
    --n_calib="$N_CALIB_DATA" \
    --submission_model_source="$SUBMISSION_MODEL_SOURCE" \
    --gpu \
    --save_cache_files \
    --output_path="$OUTPUT_PATH"

echo -e "\n============= End of Calibration ============="

##############################
# Step 4: Check Equivalence of Outputs
##############################

# echo -e "\n============= STEP 3: Check Equivalence of Outputs ============="

# python -m ci_file.backward_compatibility_test_qllama3_forward_test \
#     --model_path="$CHECKPOINT_PATH" \
#     --quant_config_path="$QUANT_CONFIG_PATH" \
#     --submission_quant_param_path="$QUANT_PARAM_PATH" \
#     --submission_quant_format_path="$QUANT_FORMAT_PATH" \
#     --n_data="$N_EVAL_DATA" \
#     --dataset_path="$DATASET_PATH" \
#     --logit_folder_path="$LOGIT_FOLDER_PATH" \
#     --gpu \
#     --generation_result_folder_path="$OUTPUT_FOLDER_PATH" \
#     --ref_path="$REF_PATH" \
#     --res_path="$RES_PATH" \
#     --config_dtype="$CONFIG_DTYPE" \
#     --update_gen_list  # Uncomment if you need to update the reference outputs

echo -e "\n============= End of Forward Test for LLaMA3.1-70B ============="

##############################
# Step 5: Cleanup and Exit
##############################

# Unset environment variables
unset SCENARIO
unset N_COUNT
unset LOG_PATH
unset N_CALIB_DATA
unset N_EVAL_DATA

# Deactivate the Conda environment
conda deactivate

# Return to the git root directory
cd "$GIT_ROOT_DIR"
