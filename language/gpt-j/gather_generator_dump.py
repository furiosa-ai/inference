import json

from gather_log_accuracy import get_args, list_sorted_subdirectories


# Save the merged data to a new file
def save_merged_data(merged_data, output_path):
    with open(output_path, "w") as f:
        json.dump(merged_data, f)


# Merge JSON data from multiple files, filtering specific keys and adjusting values
def merge_json_data(subdirectories):
    merged_data = []
    cumulative_length = 0

    for _, subdir in enumerate(subdirectories):
        json_file = subdir / "generator_dump.json"
        if json_file.exists():
            with open(json_file, "r") as f:
                eval_log = json.load(f)

                # Adjust seq_id and qsl_idx values
                for item in eval_log:
                    item["qsl_idx"] += cumulative_length

                cumulative_length += len(eval_log)
                merged_data.extend(eval_log)

    merged_data = sorted(merged_data, key=lambda x: x["qsl_idx"])
    return merged_data


# Main function
def main():
    args = get_args()
    path = args.log_dir

    subdirectories = list_sorted_subdirectories(path)
    merged_data = merge_json_data(subdirectories)

    output_path = path / f"generator_dump_n{len(merged_data)}.json"
    save_merged_data(merged_data, output_path)

    print(f"Merged JSON data saved to {output_path}")


if __name__ == "__main__":
    main()
