# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python
"""
This script processes object coverage in three dimensions (height, length, width)
based on the precomputed object visibility information and SceneInfoHandler.
For each object (excluding those with categories in NONINFORMATIVE_DESC), we aim to find
a minimal combination of images such that the union of the visible 3D points in that
dimension exactly (within a specified tolerance) covers the object’s target size.
A combination is considered minimal if removing any image from it would result in incomplete coverage.
The output for each object is stored as a dict keyed by the number of images in the combination,
e.g.:
    {
      "height": { 1: [ [img1], [img3], ... ], 2: [ [img2, img5], ... ], ... },
      "length": { ... },
      "width": { ... }
    }
Note: Theoretically, one could compute visible images on the fly by projecting the object's 3D points
into each image and checking visibility; however, precomputing and saving the object visibility data
(such as in our previous step) can save time in subsequent processing.
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import mmengine
import random
import argparse
random.seed(0)
# Tolerance for dimension coverage (10% of target)
TOLERANCE = 0.1

from spatial_engine.utils.scannet_utils.handler.info_handler import SceneInfoHandler

DEBUG = False


def load_visibility_dict(parquet_file):
    """
    Load parquet file and convert to dict with structure:
      key: "scene_id:image_to_points:image_id"
      value: parsed point index list (stored as JSON string)
    """
    df = pd.read_parquet(parquet_file)
    keys = df["key"].tolist()
    values = df["values"].tolist()
    return dict(zip(keys, values))


def compute_coverage(scene_pts, indices_bool_mask, axis):
    """
    Given a set of indices (subset of scene_pts) and the 3D points (scene_pts),
    compute the coverage along the specified axis.
    """
    if not indices_bool_mask.any():
        return None
    coords = scene_pts[indices_bool_mask][:, axis]
    return max(coords) - min(coords)


def covers_dimension(coverage, target, tolerance):
    """
    Check if the computed coverage is within tolerance of the target dimension.
    """
    if coverage is None:
        return False
    return abs(coverage - target) <= tolerance * target

def find_minimal_combinations(
    scene_id, 
    scene_pts, 
    object_points_indices, 
    visible_images, 
    images_to_visible_points_dict, 
    axis, 
    target_dim, 
    tolerance, 
    max_images=5
):
    """
    Use two-phase breadth-first search (BFS) to find "all minimal combinations" layer by layer:
      1. Phase A: Check coverage, record minimal solutions, and add uncovered combinations to expansion list;
      2. Phase B: Save minimal solutions to global set for pruning, then expand the list to the next layer.
    Once a combination covers the target, it stops expanding; but this doesn't affect other combinations
    continuing to search, thus finding all mutually exclusive minimal solutions.

    Returns a dictionary: {k: [list of minimal combinations of size k]}.
    """
    # ----------- Preprocessing -----------
    # 1) Build boolean mask for object points
    object_points_indices_mask = np.zeros(len(scene_pts), dtype=bool)
    object_points_indices_mask[object_points_indices] = True

    # 2) Extract boolean mask for each image (only keeping object points)
    image_bool_masks = {}
    for img in visible_images:
        key = f"{scene_id}:image_to_points:{img}"
        if key not in images_to_visible_points_dict:
            print(f"[Warning] Scene {scene_id}, image {img} not found in visibility dict. Skip this combination.")
            continue  # skip
        bool_mask = np.zeros(len(scene_pts), dtype=bool)
        bool_mask[json.loads(images_to_visible_points_dict[key])] = True
        bool_mask = np.logical_and(bool_mask, object_points_indices_mask)
        image_bool_masks[img] = bool_mask

    valid_images = list(image_bool_masks.keys())
    # Sort by number of covered points in descending order, optional (sometimes finds smaller combinations faster)
    # valid_images.sort(key=lambda x: np.sum(image_bool_masks[x]), reverse=True)

    # only get 25 images
    if len(valid_images) > 25:
        valid_images = random.sample(valid_images, 25)

    # Pre-compute cumulative remaining coverage, the union mask of all images from index i to the end
    cumulative_union = [None] * len(valid_images)
    if valid_images:
        cumulative_union[-1] = image_bool_masks[valid_images[-1]].copy()
        for i in range(len(valid_images) - 2, -1, -1):
            cumulative_union[i] = np.logical_or(image_bool_masks[valid_images[i]], cumulative_union[i + 1])

    # Initialize a list to store the minimal sets as boolean masks
    found_minimal_sets = []

    def is_superset_of_any_minimal_bit(comb_bitmask):
        # Check if the combination is a superset of any known minimal set using numpy operations
        if not found_minimal_sets:
            return False
        # Perform a logical AND between the combination bitmask and each minimal set
        # Then check if any row in the result matches the minimal set itself
        for minimal_set in found_minimal_sets:
            if np.array_equal(np.logical_and(minimal_set, comb_bitmask), minimal_set):
                return True
        return False

    # Helper function: Check coverage
    def can_cover(union_mask):
        cov = compute_coverage(scene_pts, union_mask, axis)
        return covers_dimension(cov, target_dim, tolerance)

    # ----------- BFS Initialization -----------
    # current_level stores (comb, union_mask, last_idx), representing all size=k combinations at the current level
    current_level = []
    for i, img in enumerate(valid_images):
        comb_bitmask = np.zeros(len(valid_images), dtype=bool)
        comb_bitmask[i] = True
        current_level.append(([img], image_bool_masks[img], i, comb_bitmask))
    
    # Store results: dictionary {combination size k: [list of minimal combinations of size k]}
    minimal_solutions = {}
    first_layer_to_expand_list = []

    k = 1
    # Enter loop while k is within limit and current_level is not empty
    while k <= max_images and current_level:
        # ---------- Phase A: Check Coverage -----------
        to_expand = []     # Combinations to expand to the next level
        new_minimal_sets = []

        # Traverse the current level
        for comb, union_mask, last_idx, comb_bitmask in current_level:
        # for comb, union_mask, last_idx, comb_bitmask in tqdm(current_level, desc=f"Processing layer {k}"):
            # Prune: Skip if the combination is a superset of any known minimal set
            if is_superset_of_any_minimal_bit(comb_bitmask):
                continue

            # If not a superset, check coverage
            if can_cover(union_mask):
                # This is a new minimal combination
                new_minimal_sets.append(comb_bitmask)
                # Save to minimal_solutions
                minimal_solutions.setdefault(k, [])
                minimal_solutions[k].append(tuple(comb))
            else:
                # Not yet covered, add to the list to expand
                # early prune: if the cumulative union is not covered, skip
                if last_idx < len(valid_images) - 1:
                    possible_union_mask = np.logical_or(cumulative_union[last_idx], union_mask)
                    if not can_cover(possible_union_mask):
                        continue
                to_expand.append((comb, union_mask, last_idx, comb_bitmask))
                if k == 1:
                    first_layer_to_expand_list.append((comb, union_mask, last_idx, comb_bitmask))

        # Update known minimal sets
        if new_minimal_sets:
            found_minimal_sets.extend(new_minimal_sets)
        
        # ---------- Phase B: Expand to Next Level -----------
        # Generate next level combinations (k+1), only expand "not covered" combinations

        next_level = []
        if k < max_images:  # If further expansion is possible
            for comb, union_mask, last_idx, comb_bitmask in to_expand:
                # Only use those in first_layer_to_expand_list to expand
                # Need to find the first larger image index
                for potential_comb, potential_union_mask, potential_last_idx, potential_comb_bitmask in first_layer_to_expand_list:
                    if potential_last_idx > last_idx:
                        # Make a new combination
                        assert len(potential_comb) == 1, f"Number of images in potential_comb should be 1. {potential_comb}."
                        new_comb = comb + potential_comb
                        new_union_mask = np.logical_or(union_mask, potential_union_mask)
                        new_comb_bitmask = np.logical_or(potential_comb_bitmask, comb_bitmask)
                        next_level.append((new_comb, new_union_mask, potential_last_idx, new_comb_bitmask))
        
        if len(next_level) > 5000:
            # random sample 5000
            next_level = random.sample(next_level, 5000)

        # Update current_level to the next level
        current_level = next_level
        k += 1

    return minimal_solutions

def process_object(scene_id, object_id, scene_info_handler: SceneInfoHandler, visible_images, images_to_visible_points_dict):
    """
    For a given object in a scene:
      - Get its point indices and corresponding 3D coordinates (aligned).
      - Use the precomputed object visibility (from the visibility file) to obtain the set of images
        that see this object.
      - For each dimension, compute the target value and corresponding axis:
            Height: target = get_object_height, axis = 2.
            For length and width, use get_object_width_axis_aligned to determine:
              if width_axis == 0 then width is along x (axis=0) and length along y (axis=1);
              else width is along y (axis=1) and length along x (axis=0).
      - Use find_minimal_combinations to obtain minimal image combinations covering the target.
      - Group the combinations by the number of images.
    Returns a dict: { "height": {n: [...], ...}, "length": {n: [...], ...}, "width": {n: [...], ...} }.
    
    Note: Theoretically, one could compute the visible images on the fly by projecting the object's 3D points
    into each image and checking visibility; however, precomputing and saving the object visibility data
    (e.g., in object_visibility.pkl) can save time in subsequent processing.
    """
    # Get full aligned scene points (first 3 coords)
    scene_pts = scene_info_handler.get_scene_points_align(scene_id)[:, :3]
    object_points_indices = scene_info_handler.get_object_point_index(scene_id, object_id)

    # Height: target from get_object_height, axis=2.
    height_target = scene_info_handler.get_object_height(scene_id, object_id)
    height_axis = 2

    # For length and width, determine axis using get_object_width_axis_aligned.
    width_axis = scene_info_handler.get_object_width_axis_aligned(scene_id, object_id)  # 0 or 1
    length_axis = 1 if width_axis == 0 else 0
    length_target = scene_info_handler.get_object_length(scene_id, object_id)
    width_target = scene_info_handler.get_object_width(scene_id, object_id)

    height_combs = find_minimal_combinations(scene_id, scene_pts, object_points_indices, visible_images, images_to_visible_points_dict, height_axis, height_target, TOLERANCE)
    # for height, length, width, we do not consider those less than 0.02 m
    length_combs = find_minimal_combinations(scene_id, scene_pts, object_points_indices, visible_images, images_to_visible_points_dict, length_axis, length_target, TOLERANCE)
    width_combs = find_minimal_combinations(scene_id, scene_pts, object_points_indices, visible_images, images_to_visible_points_dict, width_axis, width_target, TOLERANCE)

    return {
        "height": height_combs,
        "length": length_combs,
        "width": width_combs
    }

def process_scene_for_coverage(scene_id, scene_info_handler, images_to_visible_points_dict, object_visibility_dict):
    """
    Process one scene:
      For each object (excluding non-informative ones), compute the minimal image combinations
      that cover the object's height, length, and width.
    Returns: (scene_id, scene_result) where scene_result maps object_id to
             { "height": {...}, "length": {...}, "width": {...} }.
    """
    print(f"Processing scene {scene_id} for object coverage.")
    scene_result = {}
    # iterate through object_visibility_dict
    scene_object_visibility = object_visibility_dict[scene_id]["object_to_images"]
    for object_id, visibility_list in tqdm(scene_object_visibility.items(), desc=f"Processing scene {scene_id} for object coverage"):
        # * process each object
        visible_images = [img["image_id"] for img in visibility_list]
        res = process_object(scene_id, object_id, scene_info_handler, visible_images, images_to_visible_points_dict)
        if res is not None:
            scene_result[object_id] = res
    return scene_id, scene_result


def process_split_objects(split_name, scene_info_path, visibility_parquet_file, object_visibility_file, output_dir):
    """
    Process one split (train or val) for object coverage.
    For each scene, compute per-object minimal image combinations for height, length, and width.
    Save three separate result files in output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_height_file = os.path.join(output_dir, f"{split_name}_object_coverage_height.pkl")
    output_length_file = os.path.join(output_dir, f"{split_name}_object_coverage_length.pkl")
    output_width_file = os.path.join(output_dir, f"{split_name}_object_coverage_width.pkl")
    warning_file = os.path.join(output_dir, "coverage_finding_warning_objects.txt")
    with open(warning_file, "w") as wf:
        wf.write("")

    # Load necessary data once
    scene_info_handler = SceneInfoHandler(scene_info_path)
    images_to_visible_points_dict = load_visibility_dict(visibility_parquet_file)
    object_visibility_dict = mmengine.load(object_visibility_file)

    # Get all scene_ids from SceneInfoHandler in main process
    scene_ids = scene_info_handler.get_all_scene_ids()

    results_height = {}
    results_length = {}
    results_width = {}

    if DEBUG:
        scene_ids = ["scene0011_00"]

    for scene_id in tqdm(scene_ids, desc=f"Processing {split_name} scenes for object coverage"):
        scene_id, scene_result = process_scene_for_coverage(scene_id, scene_info_handler, images_to_visible_points_dict, object_visibility_dict)
        if scene_result:
            results_height[scene_id] = {}
            results_length[scene_id] = {}
            results_width[scene_id] = {}
            for object_id, res in scene_result.items():
                results_height[scene_id][object_id] = res["height"] # possible to be empty
                results_length[scene_id][object_id] = res["length"]
                results_width[scene_id][object_id] = res["width"]

    with open(output_height_file, "wb") as f:
        pickle.dump(results_height, f)
    with open(output_length_file, "wb") as f:
        pickle.dump(results_length, f)
    with open(output_width_file, "wb") as f:
        pickle.dump(results_width, f)

    print(f"Finished processing split '{split_name}' for object coverage.")
    print(f"Height coverage saved to {output_height_file}")
    print(f"Length coverage saved to {output_length_file}")
    print(f"Width coverage saved to {output_width_file}")
    print(f"Warnings saved to {warning_file}")


def main():
    parser = argparse.ArgumentParser(description="Process object coverage for a given split and scene index range.")
    parser.add_argument("--split", type=str, required=True, help="Specify the split: train or val")
    parser.add_argument("--start", type=int, required=True, help="Start scene index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End scene index (exclusive)")
    args = parser.parse_args()

    split = args.split
    start_index = args.start
    end_index = args.end

    # 配置各个split的参数（根据你实际情况修改路径）
    splits = {
        "val": {
            "scene_info_path": "data/scannet/scannet_instance_data/scenes_val_info_i_D5.pkl",
            "visibility_parquet_file": "data/scannet/scannet_instance_data/val_visibility_info_D5.parquet",
            "object_visibility_file": "evaluation_data/object_perception/object_visibility.pkl",
            "output_dir": "evaluation_data/object_perception"
        },
        "train": {
            "scene_info_path": "data/scannet/scannet_instance_data/scenes_train_info_i_D5.pkl",
            "visibility_parquet_file": "data/scannet/scannet_instance_data/train_visibility_info_D5.parquet",
            "object_visibility_file": "training_data/object_perception/object_visibility.pkl",
            "output_dir": "training_data/object_perception"
        }
    }

    if DEBUG:
        split = "val"
        start_index = 0
        end_index = 1
        # output dir should have a subdir named "debug"
        splits["val"]["output_dir"] = os.path.join(splits["val"]["output_dir"], "debug")
        splits["val"]["visibility_parquet_file"] = "data/scannet/scannet_instance_data/val_visibility_info_D5_debug.parquet"

    if split not in splits:
        raise ValueError("Invalid split. Choose train or val.")

    config = splits[split]
    # 在输出目录后加上 start_end 后缀
    config["output_dir"] = os.path.join(config["output_dir"], f"{split}_{start_index}_{end_index}")
    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"])

    scene_info_handler = SceneInfoHandler(config["scene_info_path"])
    images_to_visible_points_dict = load_visibility_dict(config["visibility_parquet_file"])
    object_visibility_dict = mmengine.load(config["object_visibility_file"])

    all_scene_ids = scene_info_handler.get_all_scene_ids()
    selected_scene_ids = all_scene_ids[start_index:end_index]
    print(f"Processing scenes from index {start_index} to {end_index} (total {len(selected_scene_ids)}) in split {split}.")

    results_height = {}
    results_length = {}
    results_width = {}

    for scene_id in tqdm(selected_scene_ids, desc=f"Processing {split} scenes"):
        scene_id, scene_result = process_scene_for_coverage(scene_id, scene_info_handler, images_to_visible_points_dict, object_visibility_dict)
        if scene_result:
            results_height[scene_id] = {}
            results_length[scene_id] = {}
            results_width[scene_id] = {}
            for object_id, res in scene_result.items():
                results_height[scene_id][object_id] = res["height"]
                results_length[scene_id][object_id] = res["length"]
                results_width[scene_id][object_id] = res["width"]

    output_height_file = os.path.join(config["output_dir"], f"{split}_object_coverage_height_{start_index}_{end_index}.pkl")
    output_length_file = os.path.join(config["output_dir"], f"{split}_object_coverage_length_{start_index}_{end_index}.pkl")
    output_width_file = os.path.join(config["output_dir"], f"{split}_object_coverage_width_{start_index}_{end_index}.pkl")

    with open(output_height_file, "wb") as f:
        pickle.dump(results_height, f)
    with open(output_length_file, "wb") as f:
        pickle.dump(results_length, f)
    with open(output_width_file, "wb") as f:
        pickle.dump(results_width, f)

    print(f"Finished processing split '{split}' for scenes {start_index} to {end_index}.")
    print(f"Height coverage saved to {output_height_file}")
    print(f"Length coverage saved to {output_length_file}")
    print(f"Width coverage saved to {output_width_file}")

if __name__ == "__main__":
    main()