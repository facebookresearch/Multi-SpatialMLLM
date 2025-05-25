# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3
"""
This script should be run after find_object_coverage.sh. It's used to merge the object coverage results of different scenes.
"""

import os
import pickle
import glob
import re

def merge_dimension(split, base_dir, dimension):
    """
    For the specified split ("train" or "val"), base_dir, and dimension,
    traverse all subdirectories starting with split_ under base_dir,
    load pkl files with filenames matching f"{split}_object_coverage_{dimension}_<start>_<end>.pkl",
    merge them and return the combined dictionary.
    """
    merged = {}
    # List all subdirectories starting with split_
    subdirs = [d for d in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(f"{split}_")]
    if not subdirs:
        print(f"No subdirectories starting with {split}_ found in {base_dir}.")
        return merged

    # Use regex to parse directory names (e.g., "train_0_10")
    pattern = re.compile(fr"{split}_(\d+)_(\d+)")
    dir_ranges = []
    for d in subdirs:
        m = pattern.match(d)
        if m:
            start = int(m.group(1))
            end = int(m.group(2))
            dir_ranges.append((d, start, end))
        else:
            print(f"Directory name format does not match requirements, skipping: {d}")
    # Sort by start value
    dir_ranges.sort(key=lambda x: x[1])
    print(f"Found {len(dir_ranges)} {split} subdirectories:")
    for d, s, e in dir_ranges:
        print(f"  {d}: {s} ~ {e}")

    # Traverse each subdirectory, looking for pkl files corresponding to the dimension
    for d, s, e in dir_ranges:
        subdir_path = os.path.join(base_dir, d)
        # Filename format: "{split}_object_coverage_{dimension}_{s}_{e}.pkl"
        pattern_file = os.path.join(subdir_path, f"{split}_object_coverage_{dimension}_*_*.pkl")
        files = glob.glob(pattern_file)
        if not files:
            print(f"No {dimension} files found in subdirectory {d}, skipping.")
            continue
        for file in files:
            print(f"Loading file: {file}")
            with open(file, "rb") as f:
                data = pickle.load(f)
            # Assume each file contains a dict with scene_id as keys
            merged.update(data)
    return merged

def merge_split(split, base_dir, output_dir):
    """
    For the specified split and base_dir, merge files for height, length, width dimensions separately,
    and save to output_dir.
    """
    dims = ["height", "length", "width"]
    merged_dict = {}
    for d in dims:
        print(f"\nMerging {split} {d} files ...")
        merged = merge_dimension(split, base_dir, d)
        merged_dict[d] = merged
        num_scene = len(merged)
        print(f"After merging {split} {d}, there are {num_scene} scene_ids in total.")
        # Save results
        output_file = os.path.join(output_dir, f"merged_{split}_object_coverage_{d}.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(merged, f)
        print(f"Saved merged results to {output_file}")
    return merged_dict

def main():
    # Set base_dir for training and validation sets (please modify according to actual paths)
    train_base = "training_data/object_perception"
    val_base = "evaluation_data/object_perception"

    # Output directory can be the same as base_dir or set separately, here we use the respective base_dir directly
    # Note: Output files will be saved to the corresponding folders
    print("Starting to merge training set files ...")
    merge_split("train", train_base, train_base)

    print("\nStarting to merge validation set files ...")
    merge_split("val", val_base, val_base)

if __name__ == "__main__":
    main()