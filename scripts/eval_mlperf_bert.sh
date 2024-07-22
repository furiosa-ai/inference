#!/bin/bash

# Function to handle cleanup on script exit
cleanup() {
    echo -e "\nTerminating all background processes...\n"
    kill $(jobs -p) 2>/dev/null
    wait
    conda deactivate
    echo -e "\nBackground processes terminated.\n"
}

# Function to print script usage
usage() {
    echo "Usage: $0 [--n_count <number>] [--n_devices <number>] [--n_parts <number>] [--offset <number>] [--dump <true/false>]"
    exit 1
}

# Trap SIGINT (Ctrl+C) to run the cleanup function
trap cleanup SIGINT

# Function to print evaluation configuration
print_eval_config() {
    printf "<<EVAL_CONFIG>>\n"
    printf "\tSCENARIO: %s\n" "$SCENARIO"
    printf "\tNPU_ARCH: %s\n" "$NPU_ARCH"
    printf "\tLLM_ENGINE_ARTIFACTS_PATH: %s\n" "$LLM_ENGINE_ARTIFACTS_PATH"
    printf "\tNUM_EVAL_DATA: %s\n" "$N_COUNT"
    printf "\tBATCH_SIZE_IN_DECODE: %s\n" "$BATCH_SIZE_IN_DECODE"
    printf "\tDEVICE: %s\n" "$DEVICE"
    printf "\t\tNUM_DEVICES: %s\n" "$N_DEVICES"
    if (( N_PARTITIONS > 1 )); then
        printf "\t\tNUM_PARTITIONS: %s\n" "$N_PARTITIONS"
        printf "\t\tPARTITION_OFFSET: %s\n" "$PARTITION_OFFSET"
    fi
    printf "\tLOG_PATH: %s\n" "$LOG_PATH"
    if [ "$DO_DUMP" = true ]; then 
        printf "\tDUMP_PATH: %s\n" "$DUMP_PATH"
    fi
}

# Function to run evaluation on a single device
run_single_device_eval() {
    echo -e "\nEvaluation on device $DEVICES\n"
    SECONDS=0
    
    LOG_PATH=$LOG_PATH \
    ML_MODEL_FILE_WITH_PATH=$MODEL_PATH \
    VOCAB_FILE=$VOCAB_PATH \
    DATASET_FILE=$DATASET_PATH \
    SKIP_VERIFY_ACCURACY=true python $work_dir/run.py --scenario=$SCENARIO \
                                                      --backend=$BACKEND \
                                                      --mlperf_conf=$MLPERF_CONF \
                                                      --max_examples=$N_COUNT \
                                                      --accuracy
    duration=$SECONDS
    printf "%d minutes and %d seconds elapsed." "$((duration / 60))" "$((duration % 60))" > "$LOG_PATH/elapsed_time.log"
    echo -e "\nAll evaluations completed.\n"
    MLPERF_ACCURACY_FILE="$LOG_PATH/mlperf_log_accuracy.json"
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --n_count) N_COUNT="$2"; shift ;;
        --n_devices) N_DEVICES="$2"; shift ;;
        --n_parts) N_PARTITIONS="$2"; shift ;;
        --offset) PARTITION_OFFSET="$2"; shift ;;
        --dump) DO_DUMP="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Define environment variables
model_name=mlperf-bert
model_dir=language/bert
git_dir=$(git rev-parse --show-toplevel)
work_dir=$git_dir/$model_dir
data_dir=$git_dir/data
log_dir=$git_dir/logs
env_name=mlperf-llm
conda_base=$($CONDA_EXE info --base)

# Ensure conda base path is set
if [ -z "$conda_base" ]; then
    echo "Error: CONDA_EXE is not set or conda is not installed."
    exit 1
fi

# Enter existing conda environment
source "$conda_base/etc/profile.d/conda.sh"
conda activate "$env_name"

# Ensure conda environment activation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment $env_name."
    exit 1
fi

# Evaluate model
echo -e "\n============= Run eval =============\n"
SCENARIO=${SCENARIO:="Offline"}
BACKEND="rngd-npu"
NPU_ARCH=${NPU_ARCH:="renegade"}
MLPERF_CONF="${MLPERF_CONF:-"$git_dir/mlperf.conf"}"
MODEL_PATH=$data_dir/models/bert/model.pytorch
VOCAB_PATH=$data_dir/models/bert/vocab.txt
DATASET_PATH=$data_dir/dataset/squad/validation/dev-v1.1.json

LOG_PATH=${LOG_PATH:="$log_dir/$model_name/$SCENARIO/$(date +%Y%m%d_%H%M%S%Z)"}
DEVICE=${DEVICE:="npu"}
DEVICE_NUM=${DEVICE_NUM:="0"}
DEVICES=${DEVICES:="$DEVICE:$DEVICE_NUM:1"}
N_COUNT=${N_COUNT:="10833"}
N_DEVICES=${N_DEVICES:="1"}
N_PARTITIONS="${N_PARTITIONS:-$N_DEVICES}"
BATCH_SIZE_IN_DECODE=${BATCH_SIZE_IN_DECODE:="1"}
DUMP_PATH=""
DO_DUMP=${DO_DUMP:=false}
SKIP_VERIFY_ACCURACY=false

if [ "$DO_DUMP" = true ]; then
    DUMP_PATH="$LOG_PATH/generator_dump_n$N_COUNT.json"
fi

export NPU_ARCH=$NPU_ARCH
export DISABLE_PROFILER=1
export LOG_PATH

# Print evaluation configuration
print_eval_config

# Run evaluation
if (( N_PARTITIONS > 1 )); then
    exit 1
else
    run_single_device_eval
fi

# Run post-evaluation tasks
if $SKIP_VERIFY_ACCURACY; then
    echo -e "Skipping accuracy evaluation."
else
    python "$work_dir/accuracy-squad.py" --vocab_file=$VOCAB_PATH \
                                         --val_data=$DATASET_PATH \
                                         --log_file=$LOG_PATH/mlperf_log_accuracy.json \
                                         --out_file=$LOG_PATH/predictions.json \
                                         --max_examples=$N_COUNT
                                         &> $LOG_PATH/accuracy_result.log
    cat "$LOG_PATH/accuracy_result.log"
    echo -e "Save eval log to $LOG_PATH"
fi

echo -e "\n============= End of eval =============\n"

# Exit from conda environment
conda deactivate
