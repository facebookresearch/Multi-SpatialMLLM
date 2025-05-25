# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
In total, with D5 scene info of ScanNet
for train, we have 82M records, with 35M nonzero records. (82654914 and 35063709)
for val, we have 24M records, with 10M nonzero records. (24117039 and 10156707)
"""

import numpy as np
from tqdm import tqdm
import torch
import mmengine

from spatial_engine.utils.scannet_utils.handler.info_handler import SceneInfoHandler 
from multiprocessing import Pool
import pandas as pd
import os

# set random seed
np.random.seed(0)

DEBUG = False # Set to True for debug mode.

def save_overlap_info(overlap_info, parquet_file):
    """
    Convert a nested dictionary into a DataFrame and save it as Parquet.
    The nested dictionary is expected in the form:
    
        overlap_info[scene_id][(image_id1, image_id2)] = {
            'overlap': ...,
            'distance': ...,
            'yaw': ...,
            'pitch': ...
        }
    """
    rows = []
    for scene_id, pair_dict in overlap_info.items():
        for (img1, img2), vals in pair_dict.items():
            rows.append({
                'scene_id':  scene_id,
                'image_id1': img1,
                'image_id2': img2,
                'overlap':   vals['overlap'],
                'distance':  vals['distance'],
                'yaw':       vals['yaw'],
                'pitch':     vals['pitch']
            })
    if not rows:
        print(f"[save_overlap_info] Nothing to save to {parquet_file}.")
        return
    df = pd.DataFrame(rows)
    df.to_parquet(parquet_file, index=False)
    print(f"[save_overlap_info] Saved {len(df)} records to {parquet_file}.")

def save_overlap_info_nonzero(overlap_info, parquet_file_nonzero):
    """
    Similar to save_overlap_info, but filter out any rows where overlap == 0.
    """
    rows = []
    for scene_id, pair_dict in overlap_info.items():
        for (img1, img2), vals in pair_dict.items():
            # Only collect if overlap != 0
            if vals['overlap'] != 0.0:
                rows.append({
                    'scene_id':  scene_id,
                    'image_id1': img1,
                    'image_id2': img2,
                    'overlap':   vals['overlap'],
                    'distance':  vals['distance'],
                    'yaw':       vals['yaw'],
                    'pitch':     vals['pitch']
                })

    if not rows:
        print(f"[save_overlap_info_nonzero] No nonzero-overlap pairs to save.")
        return

    df = pd.DataFrame(rows)
    df.to_parquet(parquet_file_nonzero, index=False)
    print(f"[save_overlap_info_nonzero] Saved {len(df)} records to {parquet_file_nonzero}.")

def extract_yaw_pitch(R):
    """
    Extract the yaw and pitch angles from a rotation matrix.
    
    :param R: A 4x4 or 3x3 matrix. If 4x4, only the top-left 3x3 is used.
    :return: A tuple (yaw, pitch) in degrees.
    """
    R3 = R[:3, :3] if R.shape == (4, 4) else R
    rotated_z_axis = R3[:, 2]
    
    # Yaw: arctan2 of (y, x)
    yaw = np.degrees(np.arctan2(rotated_z_axis[1], rotated_z_axis[0]))
    # Pitch: arcsin of z component
    pitch = np.degrees(np.arcsin(rotated_z_axis[2] / np.linalg.norm(rotated_z_axis)))
    return yaw, pitch

def calculate_camera_overlap(in_bounds_dict, image_id1, image_id2, use_cuda=False):
    """
    Calculate the percentage of overlap in the field of view of two cameras using precomputed in-bounds points.
    
    :param in_bounds_dict: Dictionary containing in-bounds information for each image ID
    :param image_id1: Image ID of the first camera
    :param image_id2: Image ID of the second camera
    :return: Percentage of overlap
    """
    in_bounds1 = in_bounds_dict[image_id1]
    in_bounds2 = in_bounds_dict[image_id2]

    if torch.cuda.is_available() and use_cuda:
        # Move data to GPU
        in_bounds1 = torch.from_numpy(in_bounds1).to('cuda')
        in_bounds2 = torch.from_numpy(in_bounds2).to('cuda')

        # Points that are visible in at least one of the cameras
        visible_points_union = torch.logical_or(in_bounds1, in_bounds2)

        # Points that are visible in both cameras
        overlap_points = torch.logical_and(in_bounds1, in_bounds2)

        # Calculate the overlap percentage
        overlap_percentage = torch.sum(overlap_points).float() / torch.sum(visible_points_union).float() * 100
        return overlap_percentage.item()  # Move result back to CPU and convert to Python float
    else:
        # Points that are visible in at least one of the cameras
        visible_points_union = np.logical_or(in_bounds1, in_bounds2)
        
        # Points that are visible in both cameras
        overlap_points = np.logical_and(in_bounds1, in_bounds2)

        # Calculate the overlap percentage
        overlap_percentage = np.sum(overlap_points) / np.sum(visible_points_union) * 100
        return overlap_percentage

def process_scene(scene_id, scene_infos: SceneInfoHandler, warning_file):
    print(f"Start processing {scene_id}.")
    image_ids = scene_infos.get_all_extrinsic_valid_image_ids(scene_id)

    # get all points in the scene
    scene_points = scene_infos.get_scene_points_align(scene_id)
    scene_points = scene_points[:, :3]

    # Precompute in-bounds points for each image
    in_bounds_dict = {}
    yaw_dict = {}
    pitch_dict = {}
    positions_dict = {}
    for image_id in image_ids:
        E = scene_infos.get_extrinsic_matrix_align(scene_id, image_id)

        # project points to camera 
        scene_points_2d, scene_points_depth = scene_infos.project_3d_point_to_image(scene_id, image_id, scene_points)
        in_bounds_points = scene_infos.check_point_visibility(scene_id, image_id, scene_points_2d, scene_points_depth) # * a True or False mask

        if np.sum(in_bounds_points) == 0: 
            with open(warning_file, 'a') as f:
                f.write(f"{scene_id}: {image_id} has no in bound points\n")

        in_bounds_dict[image_id] = in_bounds_points

        # get yaw and pitch
        yaw, pitch = extract_yaw_pitch(E)

        yaw_dict[image_id] = yaw
        pitch_dict[image_id] = pitch

        # positions
        positions_dict[image_id] = E[:3, 3]

    scene_overlap_info = {}

    for i, image_id1 in enumerate(image_ids):
        for j in range(i + 1, len(image_ids)):
            image_id2 = image_ids[j]
            overlap_percentage = calculate_camera_overlap(in_bounds_dict, image_id1, image_id2)

            yaw_diff = yaw_dict[image_id2] - yaw_dict[image_id1]
            pitch_diff = pitch_dict[image_id2] - pitch_dict[image_id1]
            distance = np.linalg.norm(positions_dict[image_id2] - positions_dict[image_id1])

            scene_overlap_info[(image_id1, image_id2)] = {}
            scene_overlap_info[(image_id1, image_id2)]['overlap'] = overlap_percentage
            scene_overlap_info[(image_id1, image_id2)]['distance'] = distance 
            scene_overlap_info[(image_id1, image_id2)]['yaw'] = yaw_diff
            scene_overlap_info[(image_id1, image_id2)]['pitch'] = pitch_diff 

            if np.any(np.isnan(list(scene_overlap_info[(image_id1, image_id2)].values()))) or \
                np.any(np.isinf(list(scene_overlap_info[(image_id1, image_id2)].values()))):
                with open(warning_file, 'a') as f:
                    f.write(f"{scene_id}: {(image_id1, image_id2)} has something wrong {list(scene_overlap_info[(image_id1, image_id2)].values())}. \n")
    
    print(f"Finished scene {scene_id}.")
    return scene_id, scene_overlap_info


def run_split(scene_info_path, output_parquet, warning_file, num_workers=15, save_interval=20):
    """
    1. Instantiate SceneInfoHandler for the given `scene_info_path`.
    2. Load existing overlap info from `output_parquet`.
    3. Process each scene in parallel and accumulate results in a dictionary.
    4. Periodically save to parquet + nonzero parquet, then final save at the end.
    """
    scene_infos = SceneInfoHandler(scene_info_path)
    overlap_info = {}

    # Gather all scenes
    all_scene_ids = scene_infos.get_all_scene_ids()
    print(f"[run_split] Found {len(all_scene_ids)} scenes in {scene_info_path}.")

    # If we're debugging, only process the first scene
    if DEBUG and len(all_scene_ids) > 1:
        all_scene_ids = all_scene_ids[:1]
        print("[run_split] DEBUG mode: processing only the first scene.")

    # Prepare arguments
    args = [(scene_id, scene_infos, warning_file) for scene_id in all_scene_ids]

    with Pool(num_workers) as pool:
        results = []
        for arg in args:
            results.append(pool.apply_async(process_scene, arg))

        for count, r in enumerate(tqdm(results, desc=f"Processing {scene_info_path}")):
            scene_id, scene_overlap_info = r.get()
            overlap_info[scene_id] = scene_overlap_info

            # Save partial results every 'save_interval' scenes
            if (count + 1) % save_interval == 0:
                save_overlap_info(overlap_info, output_parquet)

                # Also save nonzero
                nonzero_parquet = output_parquet.replace(".parquet", "_nonzero.parquet")
                save_overlap_info_nonzero(overlap_info, nonzero_parquet)

                print(f"[run_split] Saved partial results for {count + 1} scenes to {output_parquet}")

        # Final save
        save_overlap_info(overlap_info, output_parquet)
        nonzero_parquet = output_parquet.replace(".parquet", "_nonzero.parquet")
        save_overlap_info_nonzero(overlap_info, nonzero_parquet)
        print(f"[run_split] Final save to {output_parquet} complete.")
        # print total number of records
        total_records = sum(len(v) for v in overlap_info.values())
        print(f"[run_split] Total number of records: {total_records}")

        print(f"[run_split] Nonzero overlap also saved to {nonzero_parquet}.")
        # need to iterate over the overlap_info to get the number of nonzero records
        nonzero_records = sum(1 for scene in overlap_info.values() for pair in scene.values() if pair['overlap'] != 0.0)
        print(f"[run_split] Total number of nonzero records: {nonzero_records}")

def main():
    # Adjust these paths if needed
    train_info_path = "data/scannet/scannet_instance_data/scenes_train_info_i_D5.pkl"
    val_info_path   = "data/scannet/scannet_instance_data/scenes_val_info_i_D5.pkl"

    train_output_dir = "training_data/camera_movement"
    val_output_dir   = "evaluation_data/camera_movement"

    mmengine.mkdir_or_exist(train_output_dir)
    mmengine.mkdir_or_exist(val_output_dir)

    train_output_file = os.path.join(train_output_dir, "train_camera_info_D5.parquet")
    val_output_file   = os.path.join(val_output_dir,   "val_camera_info_D5.parquet")

    # Warnings
    train_warning_file = os.path.join(train_output_dir, "train_warning_D5.txt")
    val_warning_file   = os.path.join(val_output_dir,   "val_warning_D5.txt")

    # If DEBUG is true, add a suffix to file names
    if DEBUG:
        train_output_file  = train_output_file.replace(".parquet",  "_debug.parquet")
        val_output_file    = val_output_file.replace(".parquet",    "_debug.parquet")
        train_warning_file = train_warning_file.replace(".txt",     "_debug.txt")
        val_warning_file   = val_warning_file.replace(".txt",       "_debug.txt")

    num_workers = 25

    print(f"[main] DEBUG mode: {DEBUG}")
    print(f"[main] Processing train split -> {train_output_file}")
    run_split(train_info_path, train_output_file, train_warning_file, num_workers=num_workers, save_interval=20)

    print(f"[main] Processing val split -> {val_output_file}")
    run_split(val_info_path,   val_output_file,   val_warning_file,   num_workers=num_workers, save_interval=20)

if __name__ == "__main__":
    main()
