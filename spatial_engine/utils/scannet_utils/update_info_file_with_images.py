# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import mmengine
import numpy as np
from tqdm import tqdm

# Base directory where the scene_id folders are located
base_dir = "data/scannet/posed_images"
scene_infos_file = "data/scannet/scannet_instance_data/scenes_train_val_info.pkl"
frame_skip = 5 # store one image from every frame_skip images
scene_infos = mmengine.load(scene_infos_file)

# Iterate through each scene_id in the scene_info dict
for scene_id in tqdm(scene_infos.keys()):
    # Construct the path to the current scene_id folder
    scene_path = os.path.join(base_dir, scene_id)

    # Initialize the number of posed images to 0
    num_posed_images = 0

    # Initialize a dictionary to hold image data
    image_data = {}

    # Read the intrinsic matrix
    intrinsic_path = os.path.join(scene_path, "intrinsic.txt")
    with open(intrinsic_path, "r") as f:
        intrinsic_matrix = np.array(
            [list(map(float, line.split())) for line in f.readlines()]
        )

    # Iterate through each file in the scene_id directory
    all_files = os.listdir(scene_path)
    all_jpg_files = [f for f in all_files if f.endswith(".jpg")]
    all_jpg_files.sort()
    for i, filename in enumerate(all_jpg_files):
        if i % frame_skip == 0:  # Only process every frame_skip-th image
            # Extract the image_id from the filename (e.g., "00000.jpg" -> "00000")
            image_id = filename.split(".")[0]
            # Construct paths to the image, depth image, and extrinsic matrix file
            image_path = f"posed_images/{scene_id}/{filename}"
            depth_image_path = f"posed_images/{scene_id}/{image_id}.png"
            extrinsic_path = os.path.join(scene_path, f"{image_id}.txt")
            # Read the extrinsic matrix from the file
            with open(extrinsic_path, "r") as f:
                extrinsic_matrix = np.array(
                    [list(map(float, line.split())) for line in f.readlines()]
                )
            # Update the image data dictionary with this image's information
            image_data[image_id] = {
                "image_path": image_path,
                "depth_image_path": depth_image_path,
                "extrinsic_matrix": extrinsic_matrix,
            }
            # Increment the count of posed images
            num_posed_images += 1

    # Update the scene_info dictionary for the current scene_id
    scene_infos[scene_id].update(
        {
            "num_posed_images": num_posed_images,
            "images_info": image_data,
            "intrinsic_matrix": intrinsic_matrix,
        }
    )

mmengine.dump(scene_infos, scene_infos_file.replace(".pkl", f"_i_D{frame_skip}.pkl"))
