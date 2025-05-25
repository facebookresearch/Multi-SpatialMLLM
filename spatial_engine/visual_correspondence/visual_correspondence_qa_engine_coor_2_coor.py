# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
from tqdm import tqdm
import numpy as np
import random
random.seed(1)
np.random.seed(1) # * use a different seed from cam_cam
import json
import os
import mmengine
from mmengine.utils.dl_utils import TimeCounter
from spatial_engine.utils.scannet_utils.handler.info_handler import SceneInfoHandler

TASK_DESCRIPTION = [
    "Image-1: <image>\nImage-2: <image>\nGiven these two images, find the corresponding points between them. Coordinates [ x , y ] are normalized from 0 to 1 and then multiplied by 1000, with [ 0 , 0 ] starting at the top-left corner. The x-axis denotes width, and the y-axis denotes height.",
    "Image-1: <image>\nImage-2: <image>\nIdentify the matching points between these two images. The point coordinates [ x , y ] are scaled by 1000 after being normalized to a range of 0-1, originating from [ 0 , 0 ] at the top-left corner. The x-axis runs horizontally, and the y-axis runs vertically.",
    "Image-1: <image>\nImage-2: <image>\nDetermine the corresponding points in the two images provided. Coordinates [ x , y ] are normalized to a 0-1 range and scaled by 1000, starting from [ 0 , 0 ] at the top-left. The width is represented by the x-axis, and the height by the y-axis.",
    "Image-1: <image>\nImage-2: <image>\nYour task is to find the point correspondences between these images. The coordinates [ x , y ] are normalized to a 0-1 scale and then multiplied by 1000, with the origin [ 0 , 0 ] located at the top-left. The x-axis indicates width, and the y-axis indicates height.",
    "Image-1: <image>\nImage-2: <image>\nLocate the matching points in the two images. The point coordinates [ x , y ] have been normalized to a range of 0-1 and scaled by 1000, with [ 0 , 0 ] positioned at the top-left corner. The x-axis represents the width dimension, and the y-axis represents the height dimension.",
    "Image-1: <image>\nImage-2: <image>\nFind and match corresponding points across these two images. The coordinates [ x , y ] are first normalized between 0 and 1, then multiplied by 1000, where [ 0 , 0 ] is at the top-left. Width is measured along the x-axis, height along the y-axis.",
    "Image-1: <image>\nImage-2: <image>\nMap the corresponding points between the given images. Point coordinates [ x , y ] use a normalized 0-1 scale multiplied by 1000, with [ 0 , 0 ] at the top-left corner. The x-axis measures width, and the y-axis measures height.",
    "Image-1: <image>\nImage-2: <image>\nEstablish point correspondences between these two images. The [ x , y ] coordinates are normalized to 0-1 and scaled by 1000, starting at [ 0 , 0 ] in the top-left. Width corresponds to the x-axis, height to the y-axis.",
    "Image-1: <image>\nImage-2: <image>\nIdentify matching point pairs in both images. Coordinates [ x , y ] are normalized from 0 to 1, multiplied by 1000, with origin [ 0 , 0 ] at top-left. The x-axis represents width, and y-axis represents height.",
    "Image-1: <image>\nImage-2: <image>\nConnect corresponding points between the two images. The [ x , y ] coordinates use a 0-1 normalized scale multiplied by 1000, originating at [ 0 , 0 ] in the top-left. Width is on the x-axis, height on the y-axis.",
    "Image-1: <image>\nImage-2: <image>\nAnalyze and match points between these images. Point coordinates [ x , y ] are normalized to 0-1, scaled by 1000, with [ 0 , 0 ] at the top-left corner. The x-axis shows width, the y-axis shows height.",
    "Image-1: <image>\nImage-2: <image>\nLocate equivalent points in both images. The [ x , y ] coordinates are normalized between 0 and 1, multiplied by 1000, starting from [ 0 , 0 ] at top-left. Width is measured on x-axis, height on y-axis.",
    "Image-1: <image>\nImage-2: <image>\nFind matching point locations across the images. Coordinates [ x , y ] use a 0-1 normalized range scaled by 1000, with [ 0 , 0 ] at top-left. The x-axis indicates width, y-axis indicates height.",
    "Image-1: <image>\nImage-2: <image>\nIdentify corresponding point positions in these images. The [ x , y ] coordinates are normalized to 0-1 and multiplied by 1000, with origin [ 0 , 0 ] at top-left. Width follows the x-axis, height follows the y-axis.",
    "Image-1: <image>\nImage-2: <image>\nMatch points between the provided images. Point coordinates [ x , y ] are normalized from 0 to 1 and scaled by 1000, starting at [ 0 , 0 ] in the top-left. The x-axis measures width, the y-axis measures height.",
    "Image-1: <image>\nImage-2: <image>\nEstablish point-to-point correspondences in the images. The [ x , y ] coordinates use 0-1 normalization multiplied by 1000, with [ 0 , 0 ] at top-left. Width is along the x-axis, height along the y-axis.",
    "Image-1: <image>\nImage-2: <image>\nDetermine matching point locations between images. Coordinates [ x , y ] are normalized to 0-1, scaled by 1000, originating at [ 0 , 0 ] in top-left. The x-axis represents width, y-axis represents height.",
    "Image-1: <image>\nImage-2: <image>\nFind corresponding point pairs in these images. The [ x , y ] coordinates are normalized between 0 and 1, scaled by 1000, with [ 0 , 0 ] at top-left. Width is on the x-axis, height on the y-axis.",
    "Image-1: <image>\nImage-2: <image>\nLocate and match points across both images. Point coordinates [ x , y ] use a 0-1 normalized scale multiplied by 1000, starting at [ 0 , 0 ] in top-left. The x-axis shows width, y-axis shows height.",
    "Image-1: <image>\nImage-2: <image>\nIdentify equivalent point positions between images. The [ x , y ] coordinates are normalized to 0-1 and scaled by 1000, with [ 0 , 0 ] at top-left. Width follows x-axis, height follows y-axis.",
    "Image-1: <image>\nImage-2: <image>\nMap point locations between the two images. Coordinates [ x , y ] are normalized from 0 to 1, multiplied by 1000, with origin [ 0 , 0 ] at top-left. The x-axis indicates width, y-axis indicates height.",
    "Image-1: <image>\nImage-2: <image>\nFind matching points in the image pair. The [ x , y ] coordinates use 0-1 normalization scaled by 1000, starting from [ 0 , 0 ] at top-left. Width is measured along x-axis, height along y-axis.",
    "Image-1: <image>\nImage-2: <image>\nEstablish point matches between these images. Point coordinates [ x , y ] are normalized to 0-1 and multiplied by 1000, with [ 0 , 0 ] at top-left. The x-axis represents width, y-axis represents height.",
    "Image-1: <image>\nImage-2: <image>\nConnect corresponding point locations in both images. The [ x , y ] coordinates are normalized between 0 and 1, scaled by 1000, originating at [ 0 , 0 ] in top-left. Width is on x-axis, height on y-axis.",
    "Image-1: <image>\nImage-2: <image>\nIdentify matching point coordinates across images. Point positions [ x , y ] use a 0-1 normalized range multiplied by 1000, with [ 0 , 0 ] at top-left. The x-axis shows width, y-axis shows height.",
    "Image-1: <image>\nImage-2: <image>\nLocate corresponding point pairs between images. The [ x , y ] coordinates are normalized to 0-1 and scaled by 1000, starting at [ 0 , 0 ] in top-left. Width follows the x-axis, height follows the y-axis.",
    "Image-1: <image>\nImage-2: <image>\nDetermine point-to-point matches in these images. Coordinates [ x , y ] are normalized from 0 to 1, multiplied by 1000, with [ 0 , 0 ] at top-left. The x-axis measures width, y-axis measures height.",
    "Image-1: <image>\nImage-2: <image>\nFind and pair matching points between images. The [ x , y ] coordinates use 0-1 normalization scaled by 1000, with origin [ 0 , 0 ] at top-left. Width is along x-axis, height along y-axis.",
    "Image-1: <image>\nImage-2: <image>\nEstablish point correspondences across the images. Point coordinates [ x , y ] are normalized to 0-1 and multiplied by 1000, starting from [ 0 , 0 ] at top-left. The x-axis indicates width, y-axis indicates height.",
    "Image-1: <image>\nImage-2: <image>\nIdentify and match point locations between images. The [ x , y ] coordinates are normalized between 0 and 1, scaled by 1000, with [ 0 , 0 ] in top-left. Width is measured on x-axis, height on y-axis.",
    "Image-1: <image>\nImage-2: <image>\nMap corresponding points in the image pair. Point positions [ x , y ] use a 0-1 normalized scale multiplied by 1000, originating at [ 0 , 0 ] in top-left. The x-axis represents width, y-axis represents height."
]

TEMPLATES = {
    "questions": [
        "Given point coordinates [ {x1} , {y1} ] in the first image, where is this point located in the second image?",
        "Locate the equivalent position of point [ {x1} , {y1} ] from <Image-1> in <Image-2>.",
        "If a point is positioned at [ {x1} , {y1} ] in Image-1, identify its location in Image-2.",
        "Where can I find the point [ {x1} , {y1} ] from the first image in the second image?",
        "Point [ {x1} , {y1} ] is noted in <Image-1>. What are its new coordinates in <Image-2>?",
        "Transfer the point [ {x1} , {y1} ] from Image 1 to its location in Image 2.",
        "Find the position of the point [ {x1} , {y1} ] from <Image-1> when mapped onto <Image-2>.",
        "How does the point [ {x1} , {y1} ] from the first image translate in the second image?",
        "What is the equivalent of point [ {x1} , {y1} ] from Image-1 in Image-2?",
        "Identify the corresponding position of [ {x1} , {y1} ] from <Image-1> in <Image-2>.",
        "Given the coordinates [ {x1} , {y1} ] in the first image, where should it appear in the second image?",
        "If you see the point [ {x1} , {y1} ] in Image 1, where will it be located in Image 2?",
        "Determine the new position of point [ {x1} , {y1} ] from <Image-1> in <Image-2>.",
        "Where does the point [ {x1} , {y1} ] from Image-1 map to in Image-2?",
        "In the context of Image-1, what is the location of point [ {x1} , {y1} ] in Image-2?",
        "Given point [ {x1} , {y1} ] in the first image, find its coordinates in the second image.",
        "Can you find the equivalent point of [ {x1} , {y1} ] from Image 1 in Image 2?",
        "Translate the position [ {x1} , {y1} ] from <Image-1> to its new coordinates in <Image-2>.",
        "Locate the corresponding point for [ {x1} , {y1} ] from Image-1 in Image-2.",
        "If point [ {x1} , {y1} ] is found in Image 1, where is it positioned in Image 2?",
        "For the point [ {x1} , {y1} ] in Image-1, what are its coordinates in Image-2?",
        "Help me find where point [ {x1} , {y1} ] from the first image appears in the second image.",
        "Track the position [ {x1} , {y1} ] from Image 1 to its location in Image 2.",
        "What coordinates in Image-2 match the point [ {x1} , {y1} ] from Image-1?",
        "Show me where point [ {x1} , {y1} ] from <Image-1> is located in <Image-2>.",
        "Map the point [ {x1} , {y1} ] from the first image to the second image.",
        "Looking at point [ {x1} , {y1} ] in Image-1, where can it be found in Image-2?",
        "Identify where the point [ {x1} , {y1} ] from Image 1 appears in Image 2.",
        "Find the matching position in Image-2 for point [ {x1} , {y1} ] from Image-1.",
        "What is the position in Image-2 that corresponds to [ {x1} , {y1} ] in Image-1?"
    ],

    "answers": [
        "The point [ {x1} , {y1} ] in Image-1 corresponds to `[ {x2} , {y2} ]` in Image-2.",
        "In Image-2, the point equivalent to [ {x1} , {y1} ] in Image-1 is `[ {x2} , {y2} ]`.",
        "The location of the point [ {x1} , {y1} ] from Image 1 in Image 2 is `[ {x2} , {y2} ]`.",
        "The point [ {x1} , {y1} ] from <Image-1> is situated at `[ {x2} , {y2} ]` in <Image-2>.",
        "The new coordinates of the point [ {x1} , {y1} ] in Image 2 are `[ {x2} , {y2} ]`.",
        "The point [ {x1} , {y1} ] from Image-1 transfers to `[ {x2} , {y2} ]` in Image-2.",
        "When mapped onto Image-2, the point [ {x1} , {y1} ] from <Image-1> is at `[ {x2} , {y2} ]`.",
        "The point [ {x1} , {y1} ] in the first image translates to `[ {x2} , {y2} ]` in the second image.",
        "The equivalent of the point [ {x1} , {y1} ] from Image 1 in Image 2 is `[ {x2} , {y2} ]`.",
        "The corresponding position of [ {x1} , {y1} ] in Image-2 is `[ {x2} , {y2} ]`.",
        "The point [ {x1} , {y1} ] in <Image-1> is located at `[ {x2} , {y2} ]` in <Image-2>.",
        "In Image 2, the coordinates `[ {x2} , {y2} ]` correspond to the point [ {x1} , {y1} ] from Image 1.",
        "The new location of the point [ {x1} , {y1} ] from Image-1 is `[ {x2} , {y2} ]` in Image-2.",
        "For the point [ {x1} , {y1} ] from Image 1, its equivalent in Image 2 is `[ {x2} , {y2} ]`.",
        "After transformation, the point [ {x1} , {y1} ] from <Image-1> is found at `[ {x2} , {y2} ]` in <Image-2>.",
        "The relocated point [ {x1} , {y1} ] from the first image is at `[ {x2} , {y2} ]` in the second image.",
        "The coordinates in Image-2 are `[ {x2} , {y2} ]`.",
        "The position in the second image is `[ {x2} , {y2} ]`.",
        "You can find this point at `[ {x2} , {y2} ]` in Image-2.",
        "The matching location in Image-2 is `[ {x2} , {y2} ]`.",
        "The corresponding point appears at `[ {x2} , {y2} ]` in the second image.",
        "In <Image-2>, the point is located at `[ {x2} , {y2} ]`.",
        "The transformed position is `[ {x2} , {y2} ]` in Image-2.",
        "Looking at Image-2, you'll find the point at `[ {x2} , {y2} ]`.",
        "The point can be found at coordinates `[ {x2} , {y2} ]` in Image-2.",
        "In the second image, the position is `[ {x2} , {y2} ]`.",
        "The mapped location in Image-2 is `[ {x2} , {y2} ]`.",
        "You'll see this point at `[ {x2} , {y2} ]` in the second image.",
        "The point has moved to `[ {x2} , {y2} ]` in Image-2.",
        "The new position in <Image-2> is `[ {x2} , {y2} ]`."
    ]
}


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
    values = [json.loads(v) for v in tqdm(df["values"], total=len(df), desc="Converting Parquet")]

    return dict(zip(keys, values))

# --- Build QA samples based on single row overlap information ---  
def build_training_sample(scene_infos: SceneInfoHandler, row, idx: int,
                          visibility_info_dict, warning_file, max_points_per_pair=1):
    """
    Build one or more QA samples based on a row of overlap information:
      1. Get image->points mapping from visibility info to find points visible in both images;
      2. Randomly select points based on max_points_per_pair (sampling without replacement if enough points, otherwise with replacement);
      3. For each point, use scene_infos to get its 2D coordinates in both images (normalized and multiplied by 1000);
      4. If the returned 2D coordinates are empty (based on len check), print warning, write to warning file, and skip the point.
    Returns: List of constructed samples (may be empty).
    """
    scene_id = row["scene_id"]
    image1 = row["image_id1"]
    image2 = row["image_id2"]
    scene_image_height, scene_image_width = scene_infos.get_image_size(scene_id)

    # Randomly swap image1 and image2 to randomize question direction
    if random.random() < 0.5:
        image1, image2 = image2, image1

    # # Query visibility info parquet to get image_to_points for this scene_id
    # Note, if using parquet
    # if not USE_PICKLE:
        # points1 = visibility_info_dict.get(f"{scene_id}:image_to_points:{image1}", [])
        # points2 = visibility_info_dict.get(f"{scene_id}:image_to_points:{image2}", [])

    # Get information for this scene from the overall visibility info
    # else:
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

    # Sampling points: if common points count >= max_points_per_pair, sample without replacement, otherwise with replacement
    if len(common_points) >= max_points_per_pair:
        selected_points = random.sample(list(common_points), max_points_per_pair)
    else:
        selected_points = [int(random.choice(common_points.tolist())) for _ in range(max_points_per_pair)]

    conversation = []
    p1_list = []
    p2_list = []

    for pt in selected_points:
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
            continue

        x1 = round((point_2d_1[0][0] / scene_image_width) * 1000)
        y1 = round((point_2d_1[0][1] / scene_image_height) * 1000)
        x2 = round((point_2d_2[0][0] / scene_image_width) * 1000)
        y2 = round((point_2d_2[0][1] / scene_image_height) * 1000)

        task_description = random.choice(TASK_DESCRIPTION)
        question = random.choice(TEMPLATES["questions"]).format(x1=x1, y1=y1, x2=x2, y2=y2)
        answer = random.choice(TEMPLATES["answers"]).format(x1=x1, y1=y1, x2=x2, y2=y2)
        # if the first round, add task description
        if len(conversation) == 0:
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
        else:
            conversation.append({
                "from": "human",
                "value": question
            })
            conversation.append({
                "from": "gpt",
                "value": answer
            })

        p1_list.append((x1, y1))
        p2_list.append((x2, y2))

    if len(conversation) == 0:
        message = f"[build_training_sample] Warning: No conversation for scene {scene_id} {image1}, {image2}\n"
        print(message.strip())
        with open(warning_file, "a") as wf:
            wf.write(message)
        return None

    images = [f"{scene_id}/{image1}.jpg", f"{scene_id}/{image2}.jpg"]
    sample = {
        "id": f"{scene_id}_{image1}_{image2}_{idx}",
        "image": images,
        "conversations": conversation,
        "height_list": [scene_image_height, scene_image_height],
        "width_list": [scene_image_width, scene_image_width],
        "question_type": "visual_correspondence_coor_2_coor",
        "p1_list": p1_list,
        "p2_list": p2_list,
        "gt_value": list(p2_list[0])

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
    
    out_samples = []
    for idx in tqdm(range(len(df_sampled)), desc="Train"):
        row = df_sampled.iloc[idx]
        sample = build_training_sample(scene_infos, row, idx, visibility_info_dict,
                                        warning_file, max_points_per_pair=max_points_per_pair)
        if sample:
            out_samples.append(sample)
    random.shuffle(out_samples)
    out_file = os.path.join(output_dir, "train_visual_correspondence_coor_2_coor.jsonl")
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

    out_samples = []
    for idx in tqdm(range(len(df_sampled)), desc="Val"):
        row = df_sampled.iloc[idx]
        sample = build_training_sample(scene_infos, row, idx, visibility_info_dict,
                                        warning_file, max_points_per_pair=max_points_per_pair)
        if sample:
            s_eval = convert_train_sample_to_eval_sample(sample)
            out_samples.append(s_eval)
    random.shuffle(out_samples)
    out_file = os.path.join(output_dir, "val_visual_correspondence_coor_2_coor.jsonl")
    print(f"[Val] Writing {len(out_samples)} items to {out_file}")
    with open(out_file, "w") as f:
        for item in out_samples:
            f.write(json.dumps(item) + "\n")

DEBUG = False
USE_PICKLE = True 

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
        train_max_samples = 1000000
        val_max_samples = 300

    train_output_dir = os.path.join("training_data/visual_correspondence_coor_2_coor", version)
    val_output_dir = os.path.join("evaluation_data/visual_correspondence_coor_2_coor", version)
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

    build_train_dataset( # costs 4 hours to generate 1M samples
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