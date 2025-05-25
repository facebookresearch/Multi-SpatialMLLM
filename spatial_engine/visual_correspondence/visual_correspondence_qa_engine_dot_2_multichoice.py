# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
from tqdm import tqdm
import numpy as np
import random
random.seed(2)
np.random.seed(2) # * use a different seed from cam_cam
import json
import os
import mmengine
from mmengine.utils.dl_utils import TimeCounter
import cv2
from spatial_engine.utils.scannet_utils.handler.info_handler import SceneInfoHandler

DEBUG = False
USE_PICKLE = False

TASK_DESCRIPTION = [
    "Image-1: <image>\nImage-2: <image>\nGiven these two images, find the corresponding points between them.",
    "Image-1: <image>\nImage-2: <image>\nIdentify the matching points between these two images.",
    "Image-1: <image>\nImage-2: <image>\nDetermine the corresponding points in the two images provided.",
    "Image-1: <image>\nImage-2: <image>\nYour task is to find the point correspondences between these images.",
    "Image-1: <image>\nImage-2: <image>\nLocate the matching points in the two images.",
    "Image-1: <image>\nImage-2: <image>\nFind the points that correspond between these two images.",
    "Image-1: <image>\nImage-2: <image>\nMatch the points between the two images.",
    "Image-1: <image>\nImage-2: <image>\nIdentify the corresponding points in these images.",
    "Image-1: <image>\nImage-2: <image>\nDetermine which points match between the two images.",
    "Image-1: <image>\nImage-2: <image>\nFind the matching points in these images.",
    "Image-1: <image>\nImage-2: <image>\nLocate the points that correspond in the two images.",
    "Image-1: <image>\nImage-2: <image>\nIdentify the points that match between these images.",
    "Image-1: <image>\nImage-2: <image>\nDetermine the points that correspond in the images.",
    "Image-1: <image>\nImage-2: <image>\nFind the corresponding points in the two images.",
    "Image-1: <image>\nImage-2: <image>\nLocate the matching points between these images.",
    "Image-1: <image>\nImage-2: <image>\nIdentify the corresponding points in the images.",
    "Image-1: <image>\nImage-2: <image>\nDetermine which points correspond between the images.",
    "Image-1: <image>\nImage-2: <image>\nFind the points that match in these images.",
    "Image-1: <image>\nImage-2: <image>\nLocate the points that match between the images.",
    "Image-1: <image>\nImage-2: <image>\nIdentify the points that correspond in these images.",
    "Image-1: <image>\nImage-2: <image>\nDetermine the matching points in the images.",
    "Image-1: <image>\nImage-2: <image>\nFind the points that correspond between the images.",
    "Image-1: <image>\nImage-2: <image>\nLocate the corresponding points in these images.",
    "Image-1: <image>\nImage-2: <image>\nIdentify the matching points in the images.",
    "Image-1: <image>\nImage-2: <image>\nDetermine the points that match between these images.",
    "Image-1: <image>\nImage-2: <image>\nFind the corresponding points between the images.",
    "Image-1: <image>\nImage-2: <image>\nLocate the points that correspond in these images.",
    "Image-1: <image>\nImage-2: <image>\nIdentify the points that match in the images.",
    "Image-1: <image>\nImage-2: <image>\nDetermine the corresponding points between these images.",
    "Image-1: <image>\nImage-2: <image>\nFind the matching points in the images."
]

TEMPLATES = {
    "questions": [
        "Which point labeled A, B, C, or D in Image-2 corresponds to the circle point in Image-1? Please answer with the correct label from Image-2.",
        "Among the four points marked as A, B, C, and D in Image-2, which one matches the circled point in Image-1?",
        "The annotated circle in Image-1 corresponds to which of the labeled points in Image-2?",
        "Looking at Image-2, identify which of points A, B, C, or D matches the circled point from Image-1.",
        "From the four labeled points (A, B, C, D) in Image-2, which one is the corresponding point to the annotated circle in Image-1?",
        "In Image-2, there are four points labeled A, B, C, and D. Which label matches the circled point from Image-1?",
        "Find the matching point in Image-2 for the annotated circled point shown in Image-1.",
        "The point marked with a circle in Image-1 corresponds to which labeled point (A, B, C, or D) in Image-2?",
        "Identify which of the labeled points in Image-2 is the same point as the annotated circle in Image-1.",
        "Match the circled point from Image-1 to one of the four points in Image-2. What is the correct label?",
        "Which labeled point in Image-2 corresponds to the circled point in Image-1?",
        "In Image-2, which point labeled A, B, C, or D matches the circle in Image-1?",
        "Identify the point in Image-2 that corresponds to the circled point in Image-1.",
        "Which of the points labeled A, B, C, or D in Image-2 is the match for the circle in Image-1?",
        "Find the point in Image-2 that corresponds to the circle in Image-1.",
        "Which point in Image-2 matches the circled point in Image-1?",
        "Identify the matching point in Image-2 for the circle in Image-1.",
        "Which labeled point in Image-2 is the same as the circled point in Image-1?",
        "Find the corresponding point in Image-2 for the circle in Image-1.",
        "Which point in Image-2 corresponds to the circle in Image-1?",
        "Identify the point in Image-2 that matches the circle in Image-1.",
        "Which of the labeled points in Image-2 corresponds to the circle in Image-1?",
        "Find the point in Image-2 that matches the circled point in Image-1.",
        "Which point in Image-2 is the match for the circle in Image-1?",
        "Identify the corresponding point in Image-2 for the circle in Image-1.",
        "Which labeled point in Image-2 matches the circle in Image-1?",
        "Find the matching point in Image-2 for the circle in Image-1.",
        "Which point in Image-2 is the same as the circled point in Image-1?",
        "Identify the point in Image-2 that corresponds to the circle in Image-1.",
        "Which of the points in Image-2 matches the circle in Image-1?"
    ],

    "answers": [
        "The correct point is labeled `{correct_label}`.",
        "The correct point is `{correct_label}`.",
        "Point `{correct_label}` is the matching point.",
        "The point marked as `{correct_label}` corresponds to the circle.",
        "Looking at Image-2, `{correct_label}` is the matching point.",
        "The circled point corresponds to point `{correct_label}`.",
        "The corresponding point in Image-2 is labeled `{correct_label}`.",
        "Point `{correct_label}` matches the circled point from Image-1.",
        "The matching point is labeled as `{correct_label}` in Image-2.",
        "In Image-2, the point labeled `{correct_label}` is the correct match.",
        "The circle from Image-1 corresponds to point `{correct_label}` in Image-2.",
        "Point `{correct_label}` in Image-2 is the same point as the circle in Image-1.",
        "The correct label is `{correct_label}`.",
        "The point labeled `{correct_label}` is correct.",
        "Point `{correct_label}` corresponds to the circle.",
        "The correct match is point `{correct_label}`.",
        "In Image-2, `{correct_label}` is the correct point.",
        "The point `{correct_label}` is the match.",
        "The circle corresponds to point `{correct_label}`.",
        "Point `{correct_label}` is the correct match.",
        "The correct point in Image-2 is `{correct_label}`.",
        "The matching point is `{correct_label}`.",
        "The point `{correct_label}` is correct.",
        "The circle matches point `{correct_label}`.",
        "Point `{correct_label}` is the correct point.",
        "The correct match is `{correct_label}`.",
        "The point labeled `{correct_label}` is the match.",
        "The circle corresponds to `{correct_label}`.",
        "Point `{correct_label}` is the match.",
        "The correct point is `{correct_label}` in Image-2."
    ]
}

def generate_distinct_colors(n, max_retries=10):
    colors = []
    retries = 0
    while len(colors) < n and retries < max_retries:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if all(sum(abs(c1 - c2) for c1, c2 in zip(color, existing_color)) > 300 for existing_color in colors):
            colors.append(color)
        retries += 1
    if len(colors) < n:
        predefined_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0), (255, 255, 255)]  # Red, Green, Blue, Black, White
        colors += random.sample(predefined_colors, n - len(colors))
    return colors


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


@TimeCounter()
def convert_parquet_to_dict(parquet_df):
    """
    Converts a Parquet file back into the original dictionary format.

    Args:
        parquet_file (str): Path to the Parquet file.
    
    Returns:
        dict: The reconstructed visibility dictionary.
    """
    df = parquet_df 
    keys = df["key"].tolist()
    values = df["values"].tolist()

    return dict(zip(keys, values))

def build_training_sample(scene_infos: SceneInfoHandler, row, idx: int,
                          visibility_info_dict, warning_file, max_points_per_pair=1, image_output_dir="images_debug"):
    """
    Construct one or more QA samples based on a row of overlap information:
      1. Get image->points mapping from visibility info to find the set of points visible in both images;
      2. Randomly select points based on max_points_per_pair (sampling without replacement if enough points, otherwise with replacement);
      3. For each point, use scene_infos to get the 2D coordinates (normalized and multiplied by 1000) in both images;
      4. If the returned 2D coordinates are empty (based on len check), print warning, write to warning file, and skip the point.
    Returns: List of constructed samples (may be empty).
    """
    scene_id = row["scene_id"]
    image1 = row["image_id1"]
    image2 = row["image_id2"]

    # Randomly swap image1 and image2 to randomize question direction
    if random.random() < 0.5:
        image1, image2 = image2, image1

    # # Query visibility info parquet to get image_to_points for this scene_id
    # Note, if using parquet
    global USE_PICKLE
    if not USE_PICKLE:
        points1 = visibility_info_dict.get(f"{scene_id}:image_to_points:{image1}", [])
        points1 = json.loads(points1)
        points2 = visibility_info_dict.get(f"{scene_id}:image_to_points:{image2}", [])
        points2 = json.loads(points2)

    # Get information from the overall visibility info for this scene
    else:
        if scene_id not in visibility_info_dict:
            message = f"[build_training_sample] Warning: Visibility info not found for scene {scene_id}\n"
            print(message.strip())
            with open(warning_file, "a") as wf:
                wf.write(message)
            return None

        scene_visibility_info = visibility_info_dict[scene_id]
        image_to_points = scene_visibility_info.get("image_to_points", {})
        points1 = image_to_points.get(image1, [])
        points2 = image_to_points.get(image2, [])

    common_points = np.intersect1d(points1, points2)
    if len(common_points) == 0:
        message = f"[build_training_sample] Warning: No common visible points for scene {scene_id} {image1}, {image2}\n"
        print(message.strip())
        with open(warning_file, "a") as wf:
            wf.write(message)
        return None

    assert max_points_per_pair == 1, "[build_training_sample] max_points_per_pair should be 1."
    # Sampling points: if common points >= max_points_per_pair, sample without replacement, otherwise with replacement
    if len(common_points) >= max_points_per_pair:
        selected_points = random.sample(list(common_points), max_points_per_pair)
    else:
        selected_points = [int(random.choice(common_points.tolist())) for _ in range(max_points_per_pair)]

    pt = selected_points[0]

    selected_point = int(pt)
    point_2d_1 = scene_infos.get_point_2d_coordinates_in_image(
        scene_id, image1, selected_point, align=True, check_visible=True, return_depth=False)
    point_2d_2 = scene_infos.get_point_2d_coordinates_in_image(
        scene_id, image2, selected_point, align=True, check_visible=True, return_depth=False)
    # Check based on length
    if len(point_2d_1) == 0 or len(point_2d_2) == 0:
        if len(point_2d_1) == 0:
            message = f"Warning: Point {selected_point} is not visible in image {image1} in scene {scene_id}.\n"
            print(message.strip())
            with open(warning_file, "a") as wf:
                wf.write(message)
        if len(point_2d_2) == 0:
            message = f"Warning: Point {selected_point} is not visible in image {image2} in scene {scene_id}.\n"
            print(message.strip())
            with open(warning_file, "a") as wf:
                wf.write(message)
        return None 
    
    img1_path = scene_infos.get_image_path(scene_id, image1)
    img2_path = scene_infos.get_image_path(scene_id, image2)
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Randomly generate a color
    cv2.circle(img1, (int(point_2d_1[0][0]), int(point_2d_1[0][1])), 10, random_color, -1)

    correct_point = (int(point_2d_2[0][0]), int(point_2d_2[0][1]))

    # randomly generate 3 incorrect points with int coordinates, remember to check they should be different from correct_point
    incorrect_points = []
    while len(incorrect_points) < 3:
        x = random.randint(0, scene_infos.image_width - 10) # with border
        y = random.randint(0, scene_infos.image_height - 10)
        if (x, y) != correct_point:
            incorrect_points.append((x, y))

    all_points = [correct_point] + incorrect_points
    
    random.shuffle(all_points)

    labels = ['A', 'B', 'C', 'D'][:len(all_points)]  # Adjust labels to match the number of points
    random.shuffle(labels)  # Shuffle the labels

    labeled_points = {label: point for label, point in zip(labels, all_points)}
    correct_label = [label for label, point in labeled_points.items() if point == correct_point][0]

    distinct_colors = generate_distinct_colors(len(all_points))
    colors = {label: distinct_colors[i] for i, label in enumerate(labels)}

    for label, (x, y) in labeled_points.items():
        color = colors[label]
        cv2.circle(img2, (x, y), 10, color, -1)  # Increased circle radius from 5 to 10
        cv2.putText(img2, label, (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)  # Increased font scale from 0.5 to 1.0 and thickness from 1 to 2

    mmengine.mkdir_or_exist(os.path.join(image_output_dir, scene_id))
    annotated_img1_path = os.path.join(scene_id, f"{idx}_point{pt}_{image1}_{image2}_img1.jpg")
    annotated_img2_path = os.path.join(scene_id, f"{idx}_point{pt}_{image1}_{image2}_img2.jpg")
    cv2.imwrite(os.path.join(image_output_dir, annotated_img1_path), img1)
    cv2.imwrite(os.path.join(image_output_dir, annotated_img2_path), img2)        

    task_description = random.choice(TASK_DESCRIPTION)
    question = random.choice(TEMPLATES["questions"])
    answer = random.choice(TEMPLATES["answers"])

    answer = answer.format(correct_label=correct_label)

    conversation = [
        {
            "from": "human",
            "value": f"{task_description}\n{question}"
        },
        {
            "from": "gpt",
            "value": answer
        }
    ]

    sample = {
        "id": f"{idx}_p{pt}",
        "image": [annotated_img1_path, annotated_img2_path],
        "conversations": conversation,
        "height_list": [scene_infos.image_height] * 2,
        "width_list": [scene_infos.image_width] * 2,
        "question_type": "visual_correspondence_multiple_choice",
        "gt_value": correct_label,
        "p1_list": [int(point_2d_1[0][0]), int(point_2d_1[0][1])],
        "p2_list": [correct_point] + incorrect_points,
    }

    return sample

def convert_train_sample_to_eval_sample(train_sample):
    conversation = train_sample.pop("conversations")
    train_sample["text"] = conversation[0]["value"]
    return train_sample

def build_train_dataset(parquet_path, output_dir, scene_infos, desired_count,
                        overlap_min, overlap_max, interval,
                        visibility_info_path, warning_file, max_points_per_pair=1):
    df = pd.read_parquet(parquet_path)
    print(f"[Train] Loaded DataFrame with {len(df)} rows from {parquet_path}")
    print(f"[Train] Sampling {desired_count} samples with overlap in [{overlap_min}, {overlap_max}]")
    df_sampled = sample_dataframe(df, all_overlap_samples=desired_count, non_overlap_samples=0,
                                  overlap_min=overlap_min, overlap_max=overlap_max, interval=interval)
    print(f"[Train] Got {len(df_sampled)} sampled rows")

    print(f"Loading {visibility_info_path}.")
    if visibility_info_path.endswith(".parquet"):
        visibility_info_pd = pd.read_parquet(visibility_info_path)
    
        # convert the visibility_info_pd to dict
        print(f"Converting to dict.")
        visibility_info_dict = convert_parquet_to_dict(visibility_info_pd)
    else:
        visibility_info_dict = mmengine.load(visibility_info_path)
    
    image_output_dir = os.path.join(output_dir, "images")
    mmengine.mkdir_or_exist(image_output_dir)

    out_samples = []
    for idx in tqdm(range(len(df_sampled)), desc="Train"):
        row = df_sampled.iloc[idx]
        sample = build_training_sample(scene_infos, row, idx, visibility_info_dict,
                                        warning_file, max_points_per_pair=max_points_per_pair, image_output_dir=image_output_dir)
        if sample:
            out_samples.append(sample)
    random.shuffle(out_samples)
    out_file = os.path.join(output_dir, "train_visual_correspondence_dot_2_multichoice.jsonl")
    print(f"[Train] Writing {len(out_samples)} items to {out_file}")
    with open(out_file, "w") as f:
        for item in out_samples:
            f.write(json.dumps(item) + "\n")

def build_val_dataset(parquet_path, output_dir, scene_infos, desired_count,
                      overlap_min, overlap_max, interval,
                      visibility_info_path, warning_file, max_points_per_pair=1):
    assert max_points_per_pair == 1, "[Val] max_points_per_pair should be 1."
    df = pd.read_parquet(parquet_path)
    print(f"[Val] Loaded DataFrame with {len(df)} rows from {parquet_path}")
    print(f"[Val] Sampling {desired_count} samples with overlap in [{overlap_min}, {overlap_max}]")
    df_sampled = sample_dataframe(df, all_overlap_samples=desired_count, non_overlap_samples=0,
                                  overlap_min=overlap_min, overlap_max=overlap_max, interval=interval)
    print(f"[Val] Got {len(df_sampled)} sampled rows")

    print(f"Loading {visibility_info_path}.")
    if visibility_info_path.endswith(".parquet"):
        visibility_info_pd = pd.read_parquet(visibility_info_path)
    
        # convert the visibility_info_pd to dict
        print(f"Converting to dict.")
        visibility_info_dict = convert_parquet_to_dict(visibility_info_pd)
    else:
        visibility_info_dict = mmengine.load(visibility_info_path)

    image_output_dir = os.path.join(output_dir, "images")
    mmengine.mkdir_or_exist(image_output_dir)

    out_samples = []
    for idx in tqdm(range(len(df_sampled)), desc="Val"):
        row = df_sampled.iloc[idx]
        sample = build_training_sample(scene_infos, row, idx, visibility_info_dict,
                                        warning_file, max_points_per_pair=max_points_per_pair, image_output_dir=image_output_dir)
        if sample:
            s_eval = convert_train_sample_to_eval_sample(sample)
            out_samples.append(s_eval)
    random.shuffle(out_samples)
    out_file = os.path.join(output_dir, "val_visual_correspondence_dot_2_multichoice.jsonl")
    print(f"[Val] Writing {len(out_samples)} items to {out_file}")
    with open(out_file, "w") as f:
        for item in out_samples:
            f.write(json.dumps(item) + "\n")

def main():
    global USE_PICKLE
    info_path = "data/scannet/scannet_instance_data/scenes_train_val_info_i_D5.pkl"

    if USE_PICKLE:
        train_visibility_info_path = "data/scannet/scannet_instance_data/train_visibility_info_D5.pkl"
        val_visibility_info_path = "data/scannet/scannet_instance_data/val_visibility_info_D5.pkl"
    else:
        train_visibility_info_path = "data/scannet/scannet_instance_data/train_visibility_info_D5.parquet"
        val_visibility_info_path = "data/scannet/scannet_instance_data/val_visibility_info_D5.parquet" 

    overlap_min = 6
    overlap_max = 35
    interval = 1

    version = "v1_0"

    global DEBUG
    if DEBUG:
        train_parquet_path = "training_data/camera_movement/train_camera_info_D5_debug_nonzero.parquet"
        val_parquet_path   = "evaluation_data/camera_movement/val_camera_info_D5_debug_nonzero.parquet"
        train_max_samples = 100
        val_max_samples = 100
        version += "_debug"
    else:
        train_parquet_path = "training_data/camera_movement/train_camera_info_D5.parquet"
        val_parquet_path   = "evaluation_data/camera_movement/val_camera_info_D5.parquet"
        train_max_samples = 500000
        val_max_samples = 300

    train_output_dir = os.path.join("training_data/visual_correspondence_dot_2_multichoice", version)
    val_output_dir = os.path.join("evaluation_data/visual_correspondence_dot_2_multichoice", version)
    mmengine.mkdir_or_exist(train_output_dir)
    mmengine.mkdir_or_exist(val_output_dir)

    train_warning_file = os.path.join(train_output_dir, "train_warning.txt")
    val_warning_file = os.path.join(val_output_dir, "val_warning.txt")

    scene_infos = SceneInfoHandler(info_path)

    build_val_dataset(
        parquet_path=val_parquet_path,
        output_dir=val_output_dir,
        scene_infos=scene_infos,
        desired_count=val_max_samples,
        overlap_min=overlap_min,
        overlap_max=overlap_max,
        interval=interval,
        visibility_info_path=val_visibility_info_path,
        warning_file=val_warning_file,
        max_points_per_pair=1 
    )

    build_train_dataset( # takes 7 hours to generate 500K samples
        parquet_path=train_parquet_path,
        output_dir=train_output_dir,
        scene_infos=scene_infos,
        desired_count=train_max_samples,
        overlap_min=overlap_min,
        overlap_max=overlap_max,
        interval=interval,
        visibility_info_path=train_visibility_info_path,
        warning_file=train_warning_file,
        max_points_per_pair=1 
    )

if __name__ == "__main__":
    main()