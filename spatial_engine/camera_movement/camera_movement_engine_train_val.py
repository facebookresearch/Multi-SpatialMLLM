# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Multi-round QA about qualitative, quantitative, and vector of movements (distance, and two angles).

For questions, we have 30 templates. For answer and task description, we have 10 templates.
"""

import pandas as pd
from tqdm import tqdm
import numpy as np
import random
random.seed(0)
np.random.seed(0)
import json
from spatial_engine.utils.scannet_utils.handler.info_handler import SceneInfoHandler 
import os
import mmengine
import sys

sys.path.append("spatial_engine/camera_movement")
from TEMPLATES import QUESTION_TEMPLATES, ANSWER_TEMPLATES, TASK_DESCRIPTION


def sample_dataframe(df, all_overlap_samples, non_overlap_samples,
                     overlap_min=0, overlap_max=100, interval=1):
    """
    Sample from the input DataFrame, aiming to collect a total of all_overlap_samples
    from bins where (overlap != 0). If a bin doesn't have enough samples,
    the remaining quota will be passed to the next bin, and so on.

    :param df: Input DataFrame, must contain an 'overlap' column
    :param all_overlap_samples: Total desired samples from overlap>0 bins
    :param non_overlap_samples: Number of samples to take where overlap == 0
    :param overlap_min: Minimum overlap value, default is 0
    :param overlap_max: Maximum overlap value, default is 100
    :param interval: Bin interval step size, default is 1
    :return: Sampled DataFrame
    """

    # -----------------------------------------------------------
    # 1) overlap == 0: Sample these separately first
    # -----------------------------------------------------------
    non_overlap_df = df[df["overlap"] == 0].copy()
    if len(non_overlap_df) <= non_overlap_samples:
        sampled_non_overlap_df = non_overlap_df
    else:
        sampled_non_overlap_df = non_overlap_df.sample(n=non_overlap_samples)

    # Remaining data (overlap != 0)
    remaining_df = df[df["overlap"] != 0].copy()

    # -----------------------------------------------------------
    # 2) Group by overlap into bins
    # -----------------------------------------------------------
    bins = np.arange(overlap_min, overlap_max + interval, interval)
    remaining_df["overlap_group"] = pd.cut(
        remaining_df["overlap"],
        bins=bins,
        include_lowest=True
    )

    # Exclude rows not assigned to any bin (NaN)
    remaining_df = remaining_df.dropna(subset=["overlap_group"])

    # Collect data for each bin
    bin_dfs = []
    for interval_bin, group_df in remaining_df.groupby("overlap_group"):
        bin_dfs.append((interval_bin, group_df))

    if len(bin_dfs) == 0:
        # If no bins exist, return only the overlap==0 samples
        final_sampled_df = sampled_non_overlap_df
        if "overlap_group" in final_sampled_df.columns:
            final_sampled_df.drop(columns=["overlap_group"], inplace=True)
        return final_sampled_df

    # -----------------------------------------------------------
    # 3) Distribute all_overlap_samples evenly across bins
    #    and handle the remainder
    # -----------------------------------------------------------
    N = len(bin_dfs)
    base_quota = all_overlap_samples // N
    remainder = all_overlap_samples % N

    # bin_quotas[i] = base_quota, with first remainder bins getting +1
    bin_quotas = [base_quota] * N
    for i in range(remainder):
        bin_quotas[i] += 1

    # -----------------------------------------------------------
    # 4) Sort bins by data volume (small to large), not by overlap interval
    # -----------------------------------------------------------
    # Key for sorting bin_dfs: len(group_df)
    # bin_dfs[i] = (interval_bin, group_df)
    # bin_quotas[i] = base_quota +/- 1
    # To maintain one-to-one correspondence, we zip bin_dfs and bin_quotas together and sort
    bin_data = []
    for i, (interval_bin, group_df) in enumerate(bin_dfs):
        bin_data.append({
            "interval_bin": interval_bin,
            "group_df": group_df,
            "quota": bin_quotas[i],
            "size": len(group_df)  # for sorting
        })

    # Sort by size in ascending order
    bin_data.sort(key=lambda x: x["size"])

    # -----------------------------------------------------------
    # 5) Process each bin, using leftover_quota mechanism
    # -----------------------------------------------------------
    sampled_df = pd.DataFrame()
    leftover_quota = 0

    for bin_info in bin_data:
        group_df = bin_info["group_df"]
        bin_quota = bin_info["quota"]

        current_quota = bin_quota + leftover_quota
        print(f"[sample_dataframe] current_quota v.s. group_df: {current_quota} v.s. {len(group_df)}")

        if len(group_df) <= current_quota:
            # Not enough for quota, take all
            sampled_df = pd.concat([sampled_df, group_df], ignore_index=True)
            # leftover = current_quota - actual amount taken
            leftover_quota = current_quota - len(group_df)
        else:
            # More than quota, randomly sample current_quota
            sampled_part = group_df.sample(n=current_quota)
            sampled_df = pd.concat([sampled_df, sampled_part], ignore_index=True)
            leftover_quota = 0

    # If leftover_quota > 0, data is insufficient
    if leftover_quota > 0:
        print(f"[sample_dataframe] Warning: bins not enough to reach {all_overlap_samples}; leftover {leftover_quota}")

    # -----------------------------------------------------------
    # 6) Merge with overlap==0 samples
    # -----------------------------------------------------------
    final_sampled_df = pd.concat([sampled_df, sampled_non_overlap_df], ignore_index=True)

    # Cleanup
    if "overlap_group" in final_sampled_df.columns:
        final_sampled_df.drop(columns=["overlap_group"], inplace=True, errors="ignore")

    return final_sampled_df

def build_training_sample(scene_infos: SceneInfoHandler, row, idx: int, question_type: str):
    scene_id = row["scene_id"]
    image1 = row["image_id1"]
    image2 = row["image_id2"]

    overlap = float(row["overlap"])
    yaw_angle = float(row["yaw"])
    pitch_angle = float(row["pitch"])

    # randomly terminate if to swap image1 and image2
    if random.random() < 0.5:
        yaw_angle = -yaw_angle
        pitch_angle = -pitch_angle
        image1, image2 = image2, image1
    
    if abs(yaw_angle) > 180:
        if yaw_angle > 0:
            yaw_angle = yaw_angle - 360
        else:
            yaw_angle = yaw_angle + 360

    
    images = [f"{scene_id}/{image1}.jpg", f"{scene_id}/{image2}.jpg"]

    E1 = scene_infos.get_extrinsic_matrix_align(scene_id, image1) # camera to world
    E2 = scene_infos.get_extrinsic_matrix_align(scene_id, image2)

    # assert no nan
    assert not np.isnan(E1).any(), f"E1 is nan for {scene_id} {image1}"
    assert not np.isnan(E2).any(), f"E2 is nan for {scene_id} {image2}"

    # Transform E2 into the coordinate system of E1
    E1_inv = np.linalg.inv(E1)
    E2_relative = E1_inv @ E2

    # calculate the displacement vector in the first frame's coordinate system
    displacement_vector = E2_relative[:3, 3]
    distance = np.linalg.norm(displacement_vector)

    # should close to the distance from df
    assert abs(distance - row['distance']) < 0.1, f"distance is not close to the distance from df for {scene_id} {image1} {image2}."

    # the output format should be one item:
    # {"id": 1358431, "image": ["scene0006_01/01550.jpg", "scene0006_01/01245.jpg"], "conversations": [{"from": "human", "value": "Image-1: <image>\nImage-2: <image>\nAssume the scene remains unchanged. Your task is to perceive the spatial information of the scene based on the captured images. Calculate the distance (in mm) separating the cameras of images <Image-1> and <Image-2>."}, {"from": "gpt", "value": "The difference is about `1150`."}], "height_list": [968, 968], "width_list": [1296, 1296]}
    task_description = random.choice(TASK_DESCRIPTION)

    if overlap < 0.1:
        # random select from q1, q2, q3
        raise NotImplementedError("overlap < 0.1 is not supported yet.")
    else:
        question = random.choice(QUESTION_TEMPLATES[question_type])
        answer_template = random.choice(ANSWER_TEMPLATES[question_type])

        # replace the placeholder with the actual values
        # for movement, need to use > 0 to get left/right, 'forward'/'backward', 'up'/'down'
        # x -> left/right, y -> up/down, z -> forward/backward
        answer_values = {
            "x_movement": "right" if displacement_vector[0] > 0 else "left",
            "y_movement": "down" if displacement_vector[1] > 0 else "up",
            "z_movement": "forward" if displacement_vector[2] > 0 else "backward",
            "yaw_movement": "left" if yaw_angle > 0 else "right",
            "pitch_movement": "up" if pitch_angle > 0 else "down",
            "x_distance": int(abs(displacement_vector[0]) * 1000),
            "y_distance": int(abs(displacement_vector[1]) * 1000),
            "z_distance": int(abs(displacement_vector[2]) * 1000),
            "yaw_angle": int(abs(yaw_angle)),
            "pitch_angle": int(abs(pitch_angle)),
            "x_value": int(displacement_vector[0] * 1000),
            "y_value": int(displacement_vector[1] * 1000),
            "z_value": int(displacement_vector[2] * 1000),
            "total_distance": int(np.linalg.norm(displacement_vector) * 1000),
            "displacement_vector": displacement_vector.tolist(),
        }
        # map
        answer_text = answer_template.format(**answer_values)

        conversation = [
            {"from": "human", "value": f"{task_description}\n{question}"},
            {"from": "gpt", "value": answer_text},
        ]
    
    train_sample = {
        "id": idx,
        "image": images,
        "conversations": conversation,
        "height_list": [scene_infos.get_image_shape(scene_id, image1)[0]] * len(images),
        "width_list": [scene_infos.get_image_shape(scene_id, image1)[1]] * len(images),
        "answer_values": answer_values,
        "question_type": question_type,
        "gt_value": answer_values[question_type],
    }

    return train_sample

def convert_train_sample_to_eval_sample(train_sample):
    # a train sample looks like this:
    # {
    #     "id": f"{scene_id}_{object_id}_{pair_id}",
    #     "image": [f"{scene_id}/{image1}.jpg", f"{scene_id}/{image2}.jpg"],
    #     "conversations": conversation,
    #     "height_list": [image_height] * 2,
    #     "width_list": [image_width] * 2,
    #     "answer_values": answer_values,
    #     "question_type": "visual_correspondence"
    # }
    # we need a eval sample like:
    # {
    #     "id": f"{scene_id}_{object_id}_{pair_id}",
    #     "image": [f"{scene_id}/{image1}.jpg", f"{scene_id}/{image2}.jpg"],
    #     "text": question,
    #     "gt_value": value,
    #     "question_type": attr,
    # }
    conversation = train_sample.pop("conversations")
    train_sample['text'] = conversation[0]['value']

    return train_sample

def build_train_dataset(
    parquet_path,
    output_dir,
    scene_infos,
    qtype,
    desired_count,
    overlap_min,
    overlap_max,
    interval
):
    df = pd.read_parquet(parquet_path)
    print(f"[Train: {qtype}] Loaded DF with {len(df)} rows from {parquet_path}")

    print(f"[Train: {qtype}] sampling {desired_count} samples in overlap=[{overlap_min}..{overlap_max}]")

    df_sampled = sample_dataframe(
        df,
        all_overlap_samples=desired_count,
        non_overlap_samples=0,   # or your chosen number
        overlap_min=overlap_min,
        overlap_max=overlap_max,
        interval=interval
    )
    print(f"[Train: {qtype}] got {len(df_sampled)} sampled rows")

    # build samples
    out_samples = []
    for idx in tqdm(range(len(df_sampled)), desc=f"{qtype}"):
        row = df_sampled.iloc[idx]  
        s = build_training_sample(scene_infos, row, idx, qtype)
        out_samples.append(s)

    random.shuffle(out_samples)
    out_file = os.path.join(output_dir, f"{qtype}_train.jsonl")
    print(f"[Train: {qtype}] writing {len(out_samples)} items to {out_file}")
    with open(out_file, "w") as f:
        for item in out_samples:
            f.write(json.dumps(item)+"\n")

########################################################################
# Build val dataset for one question type
########################################################################

def build_val_dataset(
    parquet_path,
    output_dir,
    scene_infos,
    qtype,
    desired_count,
    overlap_min,
    overlap_max,
    interval
):
    df = pd.read_parquet(parquet_path)
    print(f"[Val: {qtype}] Loaded DF with {len(df)} rows from {parquet_path}")

    # same sampling logic
    print(f"[Val: {qtype}] sampling {desired_count} samples in overlap=[{overlap_min}..{overlap_max}]")

    df_sampled = sample_dataframe(
        df,
        all_overlap_samples=desired_count,
        non_overlap_samples=0,
        overlap_min=overlap_min,
        overlap_max=overlap_max,
        interval=interval
    )
    print(f"[Val: {qtype}] got {len(df_sampled)} sampled rows")

    # build as "train" but then convert
    out_samples = []
    for idx in tqdm(range(len(df_sampled)), desc=f"{qtype}_val"):
        row = df_sampled.iloc[idx]
        s_train = build_training_sample(scene_infos, row, idx, qtype)
        s_eval = convert_train_sample_to_eval_sample(s_train)
        out_samples.append(s_eval)

    random.shuffle(out_samples)
    out_file = os.path.join(output_dir, f"{qtype}_val.jsonl")
    print(f"[Val: {qtype}] writing {len(out_samples)} items to {out_file}")
    with open(out_file, "w") as f:
        for item in out_samples:
            f.write(json.dumps(item)+"\n")

########################################################################
# Main
########################################################################
DEBUG = False

def main():
    info_path = "data/scannet/scannet_instance_data/scenes_train_val_info_i_D5.pkl"
    overlap_min = 6
    overlap_max = 35
    interval = 1

    version = "v1_0"

    # Desired sample counts
    train_question_samples = {
        "x_movement": 1000000,
        "y_movement": 1000000,
        "z_movement": 1000000,
        "yaw_movement": 1000000,
        "pitch_movement": 1000000,
        "yaw_angle": 1000000,
        "pitch_angle": 1000000,
        "total_distance": 3000000,
        "displacement_vector": 3000000,
    }
    val_question_samples = {
        "x_movement": 300,
        "y_movement": 300,
        "z_movement": 300,
        "yaw_movement": 300,
        "pitch_movement": 300,
        "total_distance": 300,
        "yaw_angle": 300,
        "pitch_angle": 300,
        "displacement_vector": 300,
    }

    # Debug overrides
    global DEBUG
    if DEBUG:
        train_parquet_path = "training_data/camera_movement/train_camera_info_D5_debug_nonzero.parquet"
        val_parquet_path   = "evaluation_data/camera_movement/val_camera_info_D5_debug_nonzero.parquet"
        for k in train_question_samples:
            train_question_samples[k] = 100
        for k in val_question_samples:
            val_question_samples[k] = 100
        version += "_debug"
    else:
        train_parquet_path = "training_data/camera_movement/train_camera_info_D5.parquet"
        val_parquet_path   = "evaluation_data/camera_movement/val_camera_info_D5.parquet"

    train_output_dir = f"training_data/camera_movement/{version}"
    val_output_dir   = f"evaluation_data/camera_movement/{version}"

    mmengine.mkdir_or_exist(train_output_dir)
    mmengine.mkdir_or_exist(val_output_dir)

    # Initialize SceneInfoHandler
    scene_infos = SceneInfoHandler(info_path)

    # Build train & val for each question type
    for qtype in train_question_samples.keys():
        print(f"\n=== Processing question type: {qtype} ===")
        # Each type costs about 4 mins to generate 1M samples

        # Build val
        build_val_dataset(
            parquet_path=val_parquet_path,
            output_dir=val_output_dir,
            scene_infos=scene_infos,
            qtype=qtype,
            desired_count=val_question_samples[qtype],
            overlap_min=overlap_min,
            overlap_max=overlap_max,
            interval=interval
        )

        # Build train
        build_train_dataset(
            parquet_path=train_parquet_path,
            output_dir=train_output_dir,
            scene_infos=scene_infos,
            qtype=qtype,
            desired_count=train_question_samples[qtype],
            overlap_min=overlap_min,
            overlap_max=overlap_max,
            interval=interval
        )

    print("All question types processed. Done.")

if __name__ == "__main__":
    main()