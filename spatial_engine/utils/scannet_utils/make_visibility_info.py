# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import mmengine
from multiprocessing import Pool
from tqdm import tqdm
from mmengine.utils.dl_utils import TimeCounter
from spatial_engine.utils.scannet_utils.handler.info_handler import SceneInfoHandler


"""
{
  scene_id: {
    "image_to_points": {
      image_id: [point_index, ...],
      ...
    },
    "point_to_images": {
      point_index: [image_id, ...],
      ...
    }
  },
  ...
}

After generating, will convert to parquet file.
Note, at the begnining, the file is saved as pkl, so I use convert_pkl_to_parquet to convert.

"""

DEBUG = False

def convert_pkl_to_parquet(pkl_file):
    """
    Converts a previously saved PKL file to a Parquet file.

    Args:
        pkl_file (str): Path to the input PKL file.
    """
    # Load the PKL file
    import json
    import pickle
    import pandas as pd
    with open(pkl_file, 'rb') as f:
        scene_visibility_dict = pickle.load(f)
    print(f"Loaded pkl file from {pkl_file}.")

    parquet_file = pkl_file.replace(".pkl", ".parquet")
    # Convert to a format suitable for Parquet
    data = []
    for scene_id, visibility_info in scene_visibility_dict.items():
        # Process image_to_points
        for image_id, points in visibility_info["image_to_points"].items():
            key = f"{scene_id}:image_to_points:{image_id}"
            data.append((key, json.dumps(points)))  # Convert list to JSON string

        # Process point_to_images
        for point_idx, images in visibility_info["point_to_images"].items():
            key = f"{scene_id}:point_to_images:{point_idx}"
            data.append((key, json.dumps(images)))  # Convert list to JSON string

    # Create a DataFrame
    df = pd.DataFrame(data, columns=["key", "values"])

    # Save as a Parquet file
    df.to_parquet(parquet_file, index=False)

    print(f"Converted {pkl_file} to {parquet_file}. The file has {len(data)} items in total.")

def process_scene(scene_id, scene_infos, warning_file):
    """
    For one scene:
      1) Gather which points each image sees.
      2) Invert that mapping to know which images see a given point.
      3) Return scene_id and the final dict.
    """
    print(f"[process_scene] Start: {scene_id}")
    image_ids = scene_infos.get_all_extrinsic_valid_image_ids(scene_id)

    # Get all points in the scene (use only first 3 coords)
    scene_points = scene_infos.get_scene_points_align(scene_id)[:, :3]
    num_points = scene_points.shape[0]

    image_to_points = {}
    # Initially keep track of points -> images in a list of sets (index = point_id, value = set of image_ids)
    point_to_images_sets = [set() for _ in range(num_points)]

    for image_id in image_ids:
        E = scene_infos.get_extrinsic_matrix_align(scene_id, image_id)
        scene_points_2d, scene_points_depth = scene_infos.project_3d_point_to_image(
            scene_id, image_id, scene_points
        )
        in_bounds_mask = scene_infos.check_point_visibility(
            scene_id, image_id, scene_points_2d, scene_points_depth
        )

        # Record which points are visible for this image
        visible_point_indices = np.where(in_bounds_mask)[0]
        image_to_points[image_id] = visible_point_indices.tolist()

        # Also record the inverse mapping
        for idx in visible_point_indices:
            point_to_images_sets[idx].add(image_id)

        # Optional: warning if no visible points
        if len(visible_point_indices) == 0:
            with open(warning_file, 'a') as f:
                f.write(f"[Warning] {scene_id}: {image_id} has no in-bound points.\n")

    # Convert from list of sets -> dict
    point_to_images = {
        idx: sorted(list(img_set)) for idx, img_set in enumerate(point_to_images_sets) # * if one point is not observed in any images, the value is an empty list
    }

    result_dict = {
        "image_to_points": image_to_points,
        "point_to_images": point_to_images
    }
    print(f"[process_scene] Done: {scene_id}")
    return scene_id, result_dict

@TimeCounter()
def run_split(scene_info_path, output_file, warning_file, num_workers=8):
    """
    1. Loads the SceneInfoHandler for the given split.
    2. Processes each scene in parallel to get visibility info.
    3. Accumulates them into a top-level dict -> { scene_id: {...} }
    4. Saves everything into a pickle (or any other format) for easy reload.
    """
    scene_infos = SceneInfoHandler(scene_info_path)
    all_scene_ids = scene_infos.get_all_scene_ids()
    mmengine.mkdir_or_exist(os.path.dirname(output_file))

    if DEBUG and len(all_scene_ids) > 1:
        all_scene_ids = all_scene_ids[:1]
        print("[run_split] DEBUG mode. Only processing first scene.")

    print(f"[run_split] Found {len(all_scene_ids)} scenes in {scene_info_path}")
    print(f"[run_split] Output will be saved to {output_file}")

    scene_visibility_dict = {}

    # Prepare pool args
    args = [(scene_id, scene_infos, warning_file) for scene_id in all_scene_ids]

    with Pool(num_workers) as pool:
        results = [pool.apply_async(process_scene, arg) for arg in args]

        for r in tqdm(results, desc=f"Processing scenes in {scene_info_path}"):
            scene_id, visibility_info = r.get()
            scene_visibility_dict[scene_id] = visibility_info

    # Convert scene_visibility_dict to a DataFrame
    data = []
    for scene_id, visibility_info in scene_visibility_dict.items():
        # Process image_to_points
        for image_id, points in visibility_info["image_to_points"].items():
            key = f"{scene_id},image_to_points,{image_id}"
            data.append((key, points))

        # Process point_to_images
        for point_idx, images in visibility_info["point_to_images"].items():
            key = f"{scene_id},point_to_images,{point_idx}"
            data.append((key, images))

    # Create DataFrame
    df = pd.DataFrame(data, columns=["key", "values"])

    # Save to Parquet file
    df.to_parquet(output_file, index=False)

    print(f"[run_split] Done. Wrote {len(df)} entries to {output_file}")

def main():
    # Adjust these as needed
    train_info_path = "data/scannet/scannet_instance_data/scenes_train_info_i_D5.pkl"
    val_info_path   = "data/scannet/scannet_instance_data/scenes_val_info_i_D5.pkl"

    train_output_dir = "data/scannet/scannet_instance_data"
    val_output_dir   = "data/scannet/scannet_instance_data"
    mmengine.mkdir_or_exist(train_output_dir)
    mmengine.mkdir_or_exist(val_output_dir)

    # Warnings
    train_warning_file = os.path.join(train_output_dir, "make_visibility_train_warning.txt")
    val_warning_file   = os.path.join(val_output_dir,   "make_visibility_val_warning.txt")

    # Output pickle files
    train_output_file = os.path.join(train_output_dir, "train_visibility_info_D5.parquet")
    val_output_file   = os.path.join(val_output_dir,   "val_visibility_info_D5.parquet")

    # If DEBUG, tweak output to avoid overwriting real data
    global DEBUG
    if DEBUG:
        train_warning_file = train_warning_file.replace(".txt", "_debug.txt")
        val_warning_file   = val_warning_file.replace(".txt",   "_debug.txt")
        train_output_file  = train_output_file.replace(".parquet",  "_debug.parquet")
        val_output_file    = val_output_file.replace(".parquet",    "_debug.parquet")

    # Number of processes to run in parallel
    num_workers = 25

    print("[main] DEBUG =", DEBUG)

    print(f"[main] Generating val visibility -> {val_output_file}")
    run_split(val_info_path,   val_output_file,   val_warning_file,   num_workers=num_workers) # * costs 47 mins, the info file is 6G as pkl and 3.7G as parquet

    print(f"[main] Generating train visibility -> {train_output_file}")
    run_split(train_info_path, train_output_file, train_warning_file, num_workers=num_workers) # * costs 3 hours, the info file is 21G as pkl and 13G as parquet

if __name__ == "__main__":
    main()