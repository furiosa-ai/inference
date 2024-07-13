import argparse
import json
import os
from math import ceil
from pathlib import Path


def split_dataset(num_partitions, dataset_path, num_samples=None, output_dir=None):
    data_path = Path(dataset_path)

    with open(data_path, "r") as f:
        whole_data = json.load(f)

    if num_samples is not None:
        if num_samples > len(whole_data):
            raise ValueError(
                f"Number of samples to process ({num_samples}) is greater than the number of samples in the dataset ({len(whole_data)})"
            )
        if num_partitions > num_samples:
            raise ValueError(
                f"Number of partitions ({num_partitions}) is greater than the number of samples to process ({num_samples})"
            )
        whole_data = whole_data[:num_samples]

    chunk_size = ceil(len(whole_data) / num_partitions)

    if output_dir is None:
        output_dir = data_path.parent / "split"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    else:
        # Remove all files in the output directory
        for file in output_dir.iterdir():
            if file.is_file():
                file.unlink()

    for i in range(num_partitions):
        output_path = output_dir / f"split_{i}{data_path.suffix}"
        with open(output_path, "w") as f:
            json.dump(
                whole_data[i * chunk_size : i * chunk_size + chunk_size], f, indent=4
            )

        print(f"Saved split {i} to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a JSON dataset into multiple smaller files."
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
        help="Number of partitions to split the dataset into",
        default=1,
    )
    parser.add_argument("--dataset-path", type=Path, help="Path to the dataset")
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of samples to process from the dataset",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for the split files",
        default=None,
    )

    args = parser.parse_args()

    split_dataset(
        args.num_partitions, args.dataset_path, args.num_samples, args.output_dir
    )
