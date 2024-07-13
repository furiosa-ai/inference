import argparse
import json
from pathlib import Path


# Argument parser setup
def get_args():
    parser = argparse.ArgumentParser(description="Merge JSON data from subdirectories.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
        help="Directory containing log subdirectories",
    )
    return parser.parse_args()


# Create a Path object
def list_sorted_subdirectories(path):
    if not path.exists():
        print(f"The directory {path} does not exist.")
        return []

    return sorted([subdir for subdir in path.iterdir() if subdir.is_dir()])


# Merge JSON data from multiple files, filtering specific keys and adjusting values
def merge_json_data(subdirectories):
    merged_data = []
    cumulative_length = 0

    for i, subdir in enumerate(subdirectories):
        json_file = subdir / "mlperf_log_accuracy.json"
        if json_file.exists():
            with open(json_file, "r") as f:
                eval_log = json.load(f)

                # Adjust seq_id and qsl_idx values
                for item in eval_log:
                    item["seq_id"] += cumulative_length
                    item["qsl_idx"] += cumulative_length

                cumulative_length += len(eval_log)
                merged_data.extend(eval_log)

    return merged_data


# Save the merged data to a new file
def save_merged_data(merged_data, output_path):
    with open(output_path, "w") as f:
        json.dump(merged_data, f, indent=4)


# Main function
def main():
    args = get_args()
    path = args.log_dir

    subdirectories = list_sorted_subdirectories(path)
    merged_data = merge_json_data(subdirectories)

    output_path = path / "merged_mlperf_log_accuracy.json"
    save_merged_data(merged_data, output_path)

    print(f"Merged JSON data saved to {output_path}")


if __name__ == "__main__":
    main()
