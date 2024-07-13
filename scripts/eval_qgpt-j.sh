#!/bin/bash

# Function to handle cleanup on script exit
cleanup() {
  echo -e "\nTerminating all background processes...\n"
  kill $(jobs -p) 2>/dev/null  # Terminate all background processes
  wait  # Wait for background processes to terminate
  conda deactivate
  echo -e "\nBackground processes terminated.\n"
  unset model_name model_dir git_dir work_dir data_dir log_dir env_name conda_base
  unset SCENARIO BACKEND MLPERF_CONF MODEL_PATH N_DEVICES N_COUNT LOG_PATH DATASET_PATH SPLIT_DATASET_DIR DEVICE N_PARTITIONS PARTITION_OFFSET QUANT_CONFIG_PATH QUANT_PARAM_PATH QUANT_FORMAT_PATH
}

# Trap SIGINT (Ctrl+C) to run the cleanup function
trap cleanup SIGINT

# Function to print evaluation configuration
print_eval_config() {
  printf "<<EVAL_CONFIG>>\n"
  printf "\tSCENARIO: $SCENARIO\n"
  printf "\tNUM_EVAL_DATA: $N_COUNT\n"
  printf "\tDEVICE: $DEVICE\n"
  printf "\tNUM_DEVICES: $N_DEVICES\n"
  if (( N_DEVICES > 1 )); then
    printf "\tNUM_PARTITIONS: $N_PARTITIONS\n"
    printf "\tPARTITION_OFFSET: $PARTITION_OFFSET\n"
  fi
  if [ "$DO_DUMP" = true ]; then
    printf "\tDO_DUMP: true\n"
  fi
}

# Function to run evaluation on a single device
run_single_device_eval() {
  echo -e "\nEvaluation on device $DEVICE\n"
  
  if [ "$DO_DUMP" = true ]; then
    DUMP_PATH="$LOG_PATH/generator_dump_n$N_COUNT.json"
  else
    DUMP_PATH=""
  fi
  
  SECONDS=0
  LOG_PATH="$LOG_PATH" python "$work_dir/main.py" --scenario="$SCENARIO" \
                                                  --backend="$BACKEND" \
                                                  --mlperf_conf="$MLPERF_CONF" \
                                                  --model-path="$MODEL_PATH" \
                                                  --dataset-path="$DATASET_PATH" \
                                                  --max_examples="$N_COUNT" \
                                                  --gpu \
                                                  --quantize \
                                                  --quant_param_path="$QUANT_PARAM_PATH" \
                                                  --quant_format_path="$QUANT_FORMAT_PATH" \
                                                  --dump_path="$DUMP_PATH" \
                                                  --accuracy
  duration=$SECONDS
  printf "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed." > "$LOG_PATH/elapsed_time.log"
  echo -e "\nAll evaluations completed.\n"
  MLPERF_ACCURACY_FILE="$LOG_PATH/mlperf_log_accuracy.json"
}

# Function to run evaluation on multiple devices
run_multi_device_eval() {
  echo -e "\nSplit $N_COUNT dataset into $N_PARTITIONS partitions for parallel evaluation\n"
  python "$work_dir/split_dataset.py" --dataset-path="$DATASET_PATH" \
                                      --num-partitions="$N_PARTITIONS" \
                                      --num-samples="$N_COUNT" \
                                      --output-dir="$SPLIT_DATASET_DIR"
  echo -e "\nStart eval with $N_DEVICES devices\n"
  SECONDS=0
  for i in $(seq 0 $((N_DEVICES - 1))); do
    echo -e "\nEvaluation on device $DEVICE:$i\n"
    echo -e "\nPartition $((i + PARTITION_OFFSET + 1)) out of $N_PARTITIONS\n"
    DATASET_PATH_i="$SPLIT_DATASET_DIR/split_$((i + PARTITION_OFFSET)).json"
    LOG_PATH_i="$LOG_PATH/$((i + PARTITION_OFFSET))"

    if [ "$DO_DUMP" = true ]; then
      DUMP_PATH="$LOG_PATH_i/dump.json"
    else
      DUMP_PATH=""
    fi

    LOG_PATH="$LOG_PATH_i" python "$work_dir/main.py" --scenario="$SCENARIO" \
                                                      --backend="$BACKEND" \
                                                      --mlperf_conf="$MLPERF_CONF" \
                                                      --model-path="$MODEL_PATH" \
                                                      --dataset-path="$DATASET_PATH_i" \
                                                      --device="$DEVICE:$i" \
                                                      --quantize \
                                                      --quant_param_path="$QUANT_PARAM_PATH" \
                                                      --quant_format_path="$QUANT_FORMAT_PATH" \
                                                      --dump_path="$DUMP_PATH" \
                                                      --accuracy &
  done

  # Wait for all background processes to finish    
  wait
  
  duration=$SECONDS
  printf "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed." > "$LOG_PATH/elapsed_time.log"
  echo -e "\nAll evaluations completed.\n"

  # Gather log accuracy to a single file
  # If the number of devices is equal to the number of partitions, then gather log accuracy
  # Otherwise, skip the verification of accuracy since the log files are not complete
  if [ "$N_DEVICES" == "$N_PARTITIONS" ]; then
    python "$work_dir/gather_log_accuracy.py" --log-dir="$LOG_PATH"
    MLPERF_ACCURACY_FILE="$LOG_PATH/merged_mlperf_log_accuracy.json"

    python "$work_dir/generator_dump_n$N_COUNT.py" --log-dir="$LOG_PATH"
  else
    SKIP_VERIFY_ACCURACY=true
  fi
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --n_count) N_COUNT="$2"; shift ;;
    --n_devices) N_DEVICES="$2"; shift ;;
    --n_parts) N_PARTITIONS="$2"; shift ;;
    --offset) PARTITION_OFFSET="$2"; shift ;;
    --dump) DO_DUMP="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

# Define environment variables
model_name="qgpt-j"
model_dir="language/gpt-j"
git_dir=$(git rev-parse --show-toplevel)
work_dir="$git_dir/$model_dir"
data_dir="$git_dir/data"
quant_data_dir="$data_dir/quantization/gpt-j"
log_dir="$git_dir/logs"
env_name="mlperf-$model_name"
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
printf "\n============= STEP-4: Run eval =============\n"
SCENARIO="Offline"
BACKEND="${BACKEND:-rngd}"
MLPERF_CONF="${MLPERF_CONF:=$git_dir/mlperf.conf}"
MODEL_PATH="$data_dir/models/gpt-j"
N_COUNT="${N_COUNT:-13368}" # total_len=13368
LOG_PATH="${LOG_PATH:-$log_dir/$model_name/$SCENARIO/$(date +%Y%m%d_%H%M%S%Z)}"
DATASET_PATH="$data_dir/dataset/cnn-daily-mail/validation/cnn_eval.json"
SPLIT_DATASET_DIR="${SPLIT_DATASET_DIR:-$data_dir/dataset/cnn-daily-mail/validation/split}"
DEVICE="${DEVICE:-cuda}"
N_DEVICES="${N_DEVICES:-1}"
N_PARTITIONS="${N_PARTITIONS:-$N_DEVICES}"
PARTITION_OFFSET="${PARTITION_OFFSET:-0}" # Offset for selecting partition number. If 2, then evaluate on partition 2, 3, 4, ...
SKIP_VERIFY_ACCURACY=false

QUANT_CONFIG_PATH="${QUANT_CONFIG_PATH:=$quant_data_dir/quant_config.yaml}"
QUANT_PARAM_PATH="${QUANT_PARAM_PATH:=$quant_data_dir/calibration_range/quant_param.npy}"
QUANT_FORMAT_PATH="${QUANT_FORMAT_PATH:=$quant_data_dir/calibration_range/quant_format.yaml}"

DO_DUMP="${DO_DUMP:-false}"

# Print evaluation configuration
print_eval_config

# Run evaluation
if (( N_DEVICES > 1 )); then
  run_multi_device_eval
else
  run_single_device_eval
fi

# Run post-evaluation tasks
if $SKIP_VERIFY_ACCURACY; then
  echo -e "Skipping accuracy evaluation."
else
  python "$work_dir/evaluation.py" --mlperf-accuracy-file="$MLPERF_ACCURACY_FILE" \
                                   --dataset-file="$DATASET_PATH" &> "$LOG_PATH/accuracy_result.log"
  cat "$LOG_PATH/accuracy_result.log"
  echo -e "Save eval log to $LOG_PATH"
fi

echo -e "\n============= End of eval =============\n"

# Exit from conda environment
conda deactivate

# Unset environment variables
unset model_name model_dir git_dir work_dir data_dir log_dir env_name conda_base
unset SCENARIO BACKEND MLPERF_CONF MODEL_PATH N_DEVICES N_COUNT LOG_PATH DATASET_PATH SPLIT_DATASET_DIR DEVICE N_PARTITIONS PARTITION_OFFSET QUANT_CONFIG_PATH QUANT_PARAM_PATH QUANT_FORMAT_PATH
