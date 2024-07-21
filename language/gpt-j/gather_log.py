import argparse
import json
from pathlib import Path
from math import ceil

TOTAL_GPTJ_DATA_LEN = 13368

# Argument parser setup
def get_args():
    parser = argparse.ArgumentParser(description="Merge JSON data from subdirectories.")
    parser.add_argument(
        "log_dir",
        type=Path,
        help="Directory containing log subdirectories",
    )
    parser.add_argument(
        "-p",
        "--num-partitions",
        type=int,
        required=True,
        help="Number of partitions",
    )
    parser.add_argument(
        "-n",
        "--total-len",
        type=int,
        help="Total length",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if all evaluation logs are collected",
    )
    return parser.parse_args()


# Create a Path object
def list_sorted_subdirectories(path):
    if not path.exists():
        print(f"The directory {path} does not exist.")
        return []
    for subdir in path.iterdir():
        if not subdir.is_dir():
            continue
    return sorted(
        [subdir for subdir in path.iterdir() if subdir.is_dir()],
        key=lambda x: int(x.stem),
    )


# Merge JSON data from multiple files, filtering specific keys and adjusting values
def merge_json_data(subdirectories, total_len, num_parts):
    merged_data = []
    chunk_size = ceil(total_len / num_parts)
    cumulative_length = chunk_size
    
    unseen = list(range(0, num_parts))
    seen = []
    for i, subdir in enumerate(subdirectories):
        part_no = int(subdir.stem)
        if not isinstance(part_no, int):
            raise ValueError(f"Expected integer part number, got {part_no}")

        json_file = subdir / "mlperf_log_accuracy.json"
        if json_file.exists():
            try:
                with open(json_file, "r") as f:
                    eval_log = json.load(f)
                seen.append(part_no)
                unseen.remove(part_no)
            except json.decoder.JSONDecodeError:
                continue

            # Adjust seq_id and qsl_idx values
            for item in eval_log:
                if "seq_id" in item:
                    item["seq_id"] += cumulative_length * part_no
                if "qsl_idx" in item:
                    item["qsl_idx"] += cumulative_length * part_no
                if "seq_id" not in item and "qsl_idx" not in item:
                    raise ValueError(f"Expected 'seq_id' or 'qsl_idx' in item, got {item}")
            
        merged_data.extend(eval_log)

    return merged_data, seen, unseen


# Save the merged data to a new file
def save_merged_data(merged_data, output_path):
    with open(output_path, "w") as f:
        json.dump(merged_data, f, indent=4)


# Main function
def main():
    args = get_args()
    path = args.log_dir
    total_len = args.total_len
    if total_len is None:
        total_len = TOTAL_GPTJ_DATA_LEN
    num_parts = args.num_partitions

    subdirectories = list_sorted_subdirectories(path)
    merged_data, seen, unseen = merge_json_data(subdirectories, total_len, num_parts)
    n = len(merged_data)

    if args.check:
        print(f"Checking if all evaluation logs are collected...")
        print(f"- number of evaluated: {n}, # partitions: {len(seen)}")
        print(f"\tseen: {seen}")
        print(f"- number of not evaluated: {total_len - n}, # partitions: {len(unseen)}")
        print(f"\tunseen: {unseen}")

    output_path = path / f"merged_mlperf_log_accuracy.json"
    save_merged_data(merged_data, output_path)

    print(f"Merged JSON data saved to {output_path}")


if __name__ == "__main__":
    main()
