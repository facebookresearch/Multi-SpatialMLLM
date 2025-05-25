# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This script uses the saved visibility parquet file along with SceneInfoHandler to
compute the visibility of each object (excluding categories in NONINFORMATIVE_DESC) 
in all valid images for each scene.

This script needs to be run after generating the point-level visibility parquet file and before running the object frame coverage script.

The final saved result structure is:
{
    scene_id: {
        "object_to_images": {
            object_id: [
                {
                    "image_id": image_id,
                    "intersection_count": number of intersecting points,
                    "visibility": visibility percentage
                },
                ...
            ]
        },
        "image_to_objects": {
            image_id: [
                {
                    "object_id": object_id,
                    "intersection_count": number of intersecting points,
                    "visibility": visibility percentage
                },
                ...
            ]
        }
    }
}

The result saves both:
  1. object_to_images: Maps each object to a list of images that see it,
     each entry contains image_id, intersection_count and visibility percentage.
  2. image_to_objects: Maps each image to a list of objects it can see,
     also saving intersection_count and visibility.
For skipped cases, warnings are printed and written to warning.txt in the output directory.
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from spatial_engine.utils.scannet_utils.handler.info_handler import SceneInfoHandler 

# Non-informative categories (keep original strings, don't call lower())
NONINFORMATIVE_DESC = {"wall", "object", "floor", "ceiling", "window"}

def load_visibility_dict(parquet_file):
    """
    Load parquet file and convert to dict with structure:
      key: "scene_id:image_to_points:image_id"
      value: parsed point index list
    """
    df = pd.read_parquet(parquet_file)
    keys = df["key"].tolist()
    values = df["values"].tolist()

    return dict(zip(keys, values))

def process_scene(scene_id, scene_info_handler, visibility_dict):
    """
    Process a single scene:
      - Iterate through each object in the scene,
          Skip and print warning if object's raw category is in NONINFORMATIVE_DESC;
          Skip and print warning if object has no point indices.
      - For remaining objects, call get_object_point_index to get object point indices,
          Calculate threshold (5% of object points, minimum 1).
      - Iterate through all valid images (scene_info.get_all_extrinsic_valid_image_ids),
          Look up visible points for that image in visibility_dict,
          Calculate intersection with object points, record the image if intersection count meets threshold,
          Save intersection count and visibility percentage.
      - Build two mappings:
            object_to_images: { object_id: [ { "image_id": ..., "intersection_count": n, "visibility": v }, ... ] }
            image_to_objects: { image_id: [ { "object_id": ..., "intersection_count": n, "visibility": v }, ... ] }
    Returns: (scene_id, result, warnings)
    where result is dict of above two mappings, warnings is list of all warning messages for this scene.
    """
    print(f"Processing scene {scene_id}.")
    warnings_list = []
    result = {
        "object_to_images": {},
        "image_to_objects": {}
    }

    # Use the scene_info_handler
    if scene_id not in scene_info_handler.infos:
        msg = f"[Warning] Scene {scene_id} not found in scene_info."
        warnings_list.append(msg)
        print(msg)
        return scene_id, result, warnings_list

    num_objects = scene_info_handler.get_num_objects(scene_id)
    for object_id in range(num_objects):
        raw_category = scene_info_handler.get_object_raw_category(scene_id, object_id)
        if raw_category in NONINFORMATIVE_DESC:
            continue

        object_points = scene_info_handler.get_object_point_index(scene_id, object_id)
        if len(object_points) == 0:
            msg = f"[Warning] Scene {scene_id}, object {object_id} has no point indices, skipping."
            warnings_list.append(msg)
            print(msg)
            continue

        if isinstance(object_points, np.ndarray):
            object_points_set = set(object_points.tolist())
        else:
            object_points_set = set(object_points)
        total_points = len(object_points_set)
        threshold = max(1, int(0.05 * total_points))

        valid_image_ids = scene_info_handler.get_all_extrinsic_valid_image_ids(scene_id)
        for image_id in valid_image_ids:
            key = f"{scene_id}:image_to_points:{image_id}"
            if key not in visibility_dict:
                msg = f"[Warning] Scene {scene_id}, image {image_id} not found in visibility dict."
                warnings_list.append(msg)
                print(msg)
                continue

            visible_points = set(json.loads(visibility_dict[key]))
            intersection_count = len(visible_points & object_points_set)
            if intersection_count >= threshold:
                visibility_percent = (intersection_count / total_points) * 100.0
                if object_id not in result["object_to_images"]:
                    result["object_to_images"][object_id] = []
                result["object_to_images"][object_id].append({
                    "image_id": image_id,
                    "intersection_count": intersection_count,
                    "visibility": visibility_percent
                })
                if image_id not in result["image_to_objects"]:
                    result["image_to_objects"][image_id] = []
                result["image_to_objects"][image_id].append({
                    "object_id": object_id,
                    "intersection_count": intersection_count,
                    "visibility": visibility_percent
                })

    return scene_id, result, warnings_list

def process_split(split_name, scene_info_path, visibility_parquet_file, output_dir):
    """
    Process one split (train or val):
      - Load visibility parquet file and convert to dict;
      - Load scene_info to get all scene_ids;
      - Process each scene sequentially (call process_scene);
      - Save combined results to pkl file and write all warnings to warning.txt in output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_pkl_file = os.path.join(output_dir, "object_visibility.pkl")
    warning_file = os.path.join(output_dir, "warning.txt")
    with open(warning_file, "w") as wf:
        wf.write("")
    
    # Get scene_ids in the main process
    scene_info_handler = SceneInfoHandler(scene_info_path)
    scene_ids = scene_info_handler.get_all_scene_ids()
    
    results = {}
    all_warnings = []

    scene_info_handler = SceneInfoHandler(scene_info_path)
    print(f"Loading visibility dict from {visibility_parquet_file}.")
    visibility_dict = load_visibility_dict(visibility_parquet_file)
    print(f"Loaded visibility dict.")

    for scene_id in tqdm(scene_ids, desc=f"Processing {split_name} scenes"):
        scene_id, scene_result, warnings = process_scene(scene_id, scene_info_handler, visibility_dict)
        results[scene_id] = scene_result
        all_warnings.extend(warnings)

    with open(warning_file, "a") as wf:
        for w in all_warnings:
            wf.write(w + "\n")
    
    with open(output_pkl_file, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Finished processing split '{split_name}'.")
    print(f"Result saved to {output_pkl_file}")
    print(f"Warnings saved to {warning_file}")


def main():
    # Configure parameters for train and val
    splits = {
        "val": { # * take 15 mins
            "scene_info_path": "data/scannet/scannet_instance_data/scenes_val_info_i_D5.pkl",
            "visibility_parquet_file": "data/scannet/scannet_instance_data/val_visibility_info_D5.parquet",
            "output_dir": "evaluation_data/object_perception"
        },
        "train": { # * take 1h 46 mins to generate, 94M size
            "scene_info_path": "data/scannet/scannet_instance_data/scenes_train_info_i_D5.pkl",
            "visibility_parquet_file": "data/scannet/scannet_instance_data/train_visibility_info_D5.parquet",
            "output_dir": "training_data/object_perception"
        }
    }

    for split_name, config in splits.items():
        process_split(split_name,
                      config["scene_info_path"],
                      config["visibility_parquet_file"],
                      config["output_dir"])


if __name__ == "__main__":
    main()