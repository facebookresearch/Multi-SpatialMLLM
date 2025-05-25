# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import pickle
import random
from tqdm import tqdm
import numpy as np
import random
random.seed(1)
np.random.seed(1) # * use a different seed from cam_cam
import json
from spatial_engine.utils.scannet_utils.handler.info_handler import SceneInfoHandler


# ==================== Global Variables ====================
max_train_samples = -1 # -1 means no downsampling for training data; if positive, downsample to this number
val_max_samples = 3000  # Validation data will be randomly sampled to 300 samples


ASK_DESCRIPTION = [
    "Assume the scene remains unchanged. Your task is to determine the spatial properties based on the images. You need to integrate and analyze information from all provided images to get the answer.",
    "Given the static scene, determine the spatial properties using the images. Synthesize and evaluate information from all provided images to derive the answer.",
    "Analyze the images to determine the spatial properties of the scene. You must combine and interpret data from all provided images to find the answer.",
    "Using the images, identify the spatial properties of the scene. Collate and assess information from all provided images to reach the answer.",
    "Examine the images to find the spatial properties of the scene. You need to merge and scrutinize information from all provided images to obtain the answer.",
    "Determine the spatial properties of the scene based on the images. Integrate and review information from all provided images to conclude the answer.",
    "Identify the spatial properties of the scene using the images. You must gather and analyze data from all provided images to ascertain the answer.",
    "Find the spatial properties of the scene by analyzing the images. Combine and evaluate information from all provided images to deduce the answer.",
    "Use the images to determine the spatial properties of the scene. You need to synthesize and interpret information from all provided images to get the answer.",
    "Analyze the provided images to identify the spatial properties of the scene. Collate and assess data from all provided images to derive the answer.",
    "Determine the spatial properties of the scene using the images. You must integrate and scrutinize information from all provided images to find the answer.",
    "Identify the spatial properties of the scene by examining the images. Merge and evaluate data from all provided images to reach the answer.",
    "Find the spatial properties of the scene using the images. You need to gather and interpret information from all provided images to obtain the answer.",
    "Analyze the images to determine the spatial properties of the scene. Combine and review data from all provided images to ascertain the answer.",
    "Using the images, identify the spatial properties of the scene. Synthesize and scrutinize information from all provided images to deduce the answer.",
    "Examine the images to find the spatial properties of the scene. You must integrate and assess information from all provided images to get the answer.",
    "Determine the spatial properties of the scene based on the images. Collate and interpret data from all provided images to conclude the answer.",
    "Identify the spatial properties of the scene using the images. You need to merge and evaluate information from all provided images to derive the answer.",
    "Find the spatial properties of the scene by analyzing the images. Gather and review data from all provided images to find the answer.",
    "Use the images to determine the spatial properties of the scene. You must synthesize and assess information from all provided images to reach the answer.",
    "Analyze the provided images to identify the spatial properties of the scene. Integrate and interpret data from all provided images to obtain the answer.",
    "Determine the spatial properties of the scene using the images. You need to combine and scrutinize information from all provided images to ascertain the answer.",
    "Identify the spatial properties of the scene by examining the images. Collate and review data from all provided images to deduce the answer.",
    "Find the spatial properties of the scene using the images. Merge and assess information from all provided images to get the answer.",
    "Analyze the images to determine the spatial properties of the scene. You must gather and interpret data from all provided images to conclude the answer.",
    "Using the images, identify the spatial properties of the scene. Combine and evaluate information from all provided images to derive the answer.",
    "Examine the images to find the spatial properties of the scene. You need to synthesize and review data from all provided images to find the answer.",
    "Determine the spatial properties of the scene based on the images. Integrate and scrutinize information from all provided images to reach the answer.",
    "Identify the spatial properties of the scene using the images. Collate and interpret data from all provided images to obtain the answer.",
    "Find the spatial properties of the scene by analyzing the images. You must merge and assess information from all provided images to ascertain the answer."
]

QUESTION_TEMPLATES = [
    "What is the {dimension} (in millimeters) of the {object_category} itself commonly visible in these images?",
    "Calculate the {dimension} (in millimeters) of the {object_category} that is commonly visible in these images.",
    "Determine the {dimension} (in millimeters) of the {object_category} which is commonly visible in these images.",
    "Find the {dimension} (in millimeters) of the {object_category} that is commonly visible in these images.",
    "Estimate the {dimension} (in millimeters) of the {object_category} itself commonly visible in these images.",
    "Measure the {dimension} (in millimeters) of the {object_category} that is commonly visible in these images.",
    "Could you tell me the {dimension} (in millimeters) of the {object_category} which is commonly visible in these images?",
    "Please compute the {dimension} (in millimeters) of the {object_category} itself commonly visible in these images.",
    "What is the approximate {dimension} (in millimeters) of the {object_category} that is commonly visible in these images?",
    "Give the {dimension} (in millimeters) of the {object_category} which is commonly visible in these images.",
    "What is the {dimension} (in millimeters) of the {object_category} that is commonly visible across these images?",
    "Calculate the {dimension} (in millimeters) of the {object_category} itself commonly visible across these images.",
    "Determine the {dimension} (in millimeters) of the {object_category} which is commonly visible across these images.",
    "Find the {dimension} (in millimeters) of the {object_category} that is commonly visible across these images.",
    "Estimate the {dimension} (in millimeters) of the {object_category} itself commonly visible across these images.",
    "Measure the {dimension} (in millimeters) of the {object_category} that is commonly visible across these images.",
    "Could you tell me the {dimension} (in millimeters) of the {object_category} which is commonly visible across these images?",
    "Please compute the {dimension} (in millimeters) of the {object_category} itself commonly visible across these images.",
    "What is the approximate {dimension} (in millimeters) of the {object_category} that is commonly visible across these images?",
    "Give the {dimension} (in millimeters) of the {object_category} which is commonly visible across these images.",
    "What is the {dimension} (in millimeters) of the {object_category} itself that is commonly visible in these images?",
    "Calculate the {dimension} (in millimeters) of the {object_category} which is commonly visible in these images.",
    "Determine the {dimension} (in millimeters) of the {object_category} itself that is commonly visible in these images.",
    "Find the {dimension} (in millimeters) of the {object_category} which is commonly visible in these images.",
    "Estimate the {dimension} (in millimeters) of the {object_category} that is commonly visible in these images.",
    "Measure the {dimension} (in millimeters) of the {object_category} itself that is commonly visible in these images.",
    "Could you tell me the {dimension} (in millimeters) of the {object_category} that is commonly visible in these images?",
    "Please compute the {dimension} (in millimeters) of the {object_category} which is commonly visible in these images.",
    "What is the approximate {dimension} (in millimeters) of the {object_category} itself that is commonly visible in these images?",
    "Give the {dimension} (in millimeters) of the {object_category} that is commonly visible in these images."
]

ANSWER_TEMPLATES = [
    "The {dimension} is approximately `{value_mm}` millimeters.",
    "It measures about `{value_mm}` millimeters in {dimension}.",
    "I estimate the {dimension} to be around `{value_mm}` millimeters.",
    "The {object_category}'s {dimension} is roughly `{value_mm}` millimeters.",
    "Based on the images, the {dimension} is near `{value_mm}` millimeters.",
    "It appears that the {dimension} is `{value_mm}` millimeters.",
    "From my estimation, the {dimension} is `{value_mm}` millimeters.",
    "The {dimension} seems to be around `{value_mm}` millimeters.",
    "Approximately, the {dimension} is `{value_mm}` millimeters.",
    "I would say the {dimension} is `{value_mm}` millimeters.",
    "The {dimension} is estimated to be `{value_mm}` millimeters.",
    "In my view, the {dimension} is about `{value_mm}` millimeters.",
    "The {dimension} is likely around `{value_mm}` millimeters.",
    "Judging by the images, the {dimension} is approximately `{value_mm}` millimeters.",
    "The {dimension} is calculated to be `{value_mm}` millimeters.",
    "It looks like the {dimension} is `{value_mm}` millimeters.",
    "The {dimension} is assessed to be `{value_mm}` millimeters.",
    "The {dimension} is gauged at `{value_mm}` millimeters.",
    "The {dimension} is reckoned to be `{value_mm}` millimeters.",
    "The {dimension} is figured to be `{value_mm}` millimeters.",
    "The {dimension} is computed to be `{value_mm}` millimeters.",
    "The {dimension} is deduced to be `{value_mm}` millimeters.",
    "The {dimension} is inferred to be `{value_mm}` millimeters.",
    "The {dimension} is surmised to be `{value_mm}` millimeters.",
    "The {dimension} is supposed to be `{value_mm}` millimeters.",
    "The {dimension} is thought to be `{value_mm}` millimeters.",
    "The {dimension} is understood to be `{value_mm}` millimeters.",
    "The {dimension} is viewed as `{value_mm}` millimeters.",
    "The {dimension} is approximated to be `{value_mm}` millimeters based on the data.",
    "After analyzing the images, the {dimension} is concluded to be `{value_mm}` millimeters."
]

def convert_train_sample_to_eval_sample(train_sample):
    conversation = train_sample.pop("conversations")
    train_sample["text"] = conversation[0]["value"]
    return train_sample

def build_lwh_qa_samples(scene_info_handler, dimension_info_path, dimension_name, split, output_dir, max_k=6, max_samples=-1):
    """
    Construct QA samples based on merged info files, and write to different jsonl files according to combination size K.
    
    Example file structure (info file, dict saved with pickle):
      {
         scene_id: {
             object_id: {
                  "1": [ [img1], [img3], ... ],
                  "2": [ [img2, img5], ... ],
                  ...
             },
             ...
         },
         ...
      }
    
    For each combination sample, construct a training sample:
      {
         "id": "<scene_id>_<object_id>_<k>_<combo_idx>",
         "image": [ "scene_id/img1.jpg", "scene_id/img2.jpg", ... ],
         "conversations": [
              {"from": "human", "value": "<prefix lines>\n<task description>\n<question>"},
              {"from": "gpt", "value": "<answer>"}
         ],
         "height_list": [scene_info_handler.image_height, ...],  // repeated len(combo) times
         "width_list": [scene_info_handler.image_width, ...],
         "question_type": "{dimension}_estimation",
         "gt_value": <value in mm>
      }
    """
    print(f"Processing dimension: {dimension_name}, split: {split}")
    with open(dimension_info_path, "rb") as f:
        dim_info = pickle.load(f)
    os.makedirs(output_dir, exist_ok=True)

    samples_by_k = {k: [] for k in range(1, max_k+1)}

    for scene_id, obj_dict in tqdm(dim_info.items(), desc=f"Processing {dimension_name} info"):
        for object_id, k_dict in obj_dict.items():
            if dimension_name == "height":
                val_m = scene_info_handler.get_object_height(scene_id, object_id)
            elif dimension_name == "length":
                val_m = scene_info_handler.get_object_length(scene_id, object_id)
            elif dimension_name == "width":
                val_m = scene_info_handler.get_object_width(scene_id, object_id)
            else:
                val_m = 0.0
            val_mm = int(round(val_m * 1000))
            object_category = scene_info_handler.get_object_raw_category(scene_id, object_id)
            for k_str, combos in k_dict.items():
                try:
                    k_val = int(k_str)
                except:
                    continue
                if k_val < 1 or k_val > max_k:
                    continue
                for combo_idx, combo in enumerate(combos):
                    if not combo:
                        continue
                    combo = list(combo)
                    random.shuffle(combo)
                    prefix_lines = [f"Image-{i}: <image>" for i in range(1, len(combo)+1)]
                    prefix = "\n".join(prefix_lines)
                    task_line = random.choice(TASK_DESCRIPTION)
                    q_template = random.choice(QUESTION_TEMPLATES)
                    question = q_template.format(dimension=dimension_name, object_category=object_category)
                    full_question = f"{prefix}\n{task_line}\n{question}"
                    a_template = random.choice(ANSWER_TEMPLATES)
                    answer = a_template.format(dimension=dimension_name, value_mm=val_mm, object_category=object_category)
                    conversation = [
                        {"from": "human", "value": full_question},
                        {"from": "gpt", "value": answer}
                    ]
                    sample = {
                        "id": f"{scene_id}_{object_id}_{k_val}_{combo_idx}",
                        "image": [f"{scene_id}/{img}.jpg" for img in combo],
                        "conversations": conversation,
                        "height_list": [scene_info_handler.image_height] * len(combo),
                        "width_list": [scene_info_handler.image_width] * len(combo),
                        "question_type": f"object_perception_{dimension_name}_estimation",
                        "gt_value": val_mm
                    }
                    samples_by_k[k_val].append(sample)

    for k in range(1, max_k+1):
        # only consider those have at least one sample
        if len(samples_by_k[k]) == 0:
            continue
        if max_samples > 0 and len(samples_by_k[k]) > max_samples:
            samples_by_k[k] = random.sample(samples_by_k[k], max_samples)
        fname = f"object_perception_{dimension_name}_k{k}_{split}_{max_samples}.jsonl"
        fpath = os.path.join(output_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            for sample in samples_by_k[k]:
                f.write(json.dumps(sample) + "\n")
        print(f"Written K={k} {len(samples_by_k[k])} samples to {fpath}")

    print(f"Finished building QA samples for {dimension_name}.")

def build_train_and_val_datasets():
    scene_info_path = "data/scannet/scannet_instance_data/scenes_train_val_info_i_D5.pkl"
    
    train_height_info = "training_data/object_perception/merged_train_object_coverage_height.pkl"
    train_length_info = "training_data/object_perception/merged_train_object_coverage_length.pkl"
    train_width_info = "training_data/object_perception/merged_train_object_coverage_width.pkl"

    val_height_info = "evaluation_data/object_perception/merged_val_object_coverage_height.pkl"
    val_length_info = "evaluation_data/object_perception/merged_val_object_coverage_length.pkl"
    val_width_info = "evaluation_data/object_perception/merged_val_object_coverage_width.pkl"

    train_output_dir = "training_data/object_perception"
    val_output_dir = "evaluation_data/object_perception"
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(val_output_dir, exist_ok=True)

    scene_info_handler = SceneInfoHandler(scene_info_path)

    print("\nBuilding TRAIN samples ...")
    build_lwh_qa_samples(scene_info_handler, train_height_info, "height", "train", train_output_dir, max_k=6, max_samples=max_train_samples) # will generate K=1, 281335 samples, K=2 457409 samples
    build_lwh_qa_samples(scene_info_handler, train_length_info, "length", "train", train_output_dir, max_k=6, max_samples=max_train_samples) # will generate K=1, 274175 samples, K=2 667299 samples
    build_lwh_qa_samples(scene_info_handler, train_width_info, "width", "train", train_output_dir, max_k=6, max_samples=max_train_samples) # will generate K=1 256017 samples, K=2 425229 samples

    temp_val_dir = os.path.join(val_output_dir, "temp")
    os.makedirs(temp_val_dir, exist_ok=True)
    print("\nBuilding VAL samples (train format) ...")
    build_lwh_qa_samples(scene_info_handler, val_height_info, "height", "val", temp_val_dir, max_k=6, max_samples=val_max_samples)
    build_lwh_qa_samples(scene_info_handler, val_length_info, "length", "val", temp_val_dir, max_k=6, max_samples=val_max_samples)
    build_lwh_qa_samples(scene_info_handler, val_width_info, "width", "val", temp_val_dir, max_k=6, max_samples=val_max_samples)
    for fname in os.listdir(temp_val_dir):
        temp_path = os.path.join(temp_val_dir, fname)
        output_path = os.path.join(val_output_dir, fname)
        with open(temp_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                sample = json.loads(line)
                sample = convert_train_sample_to_eval_sample(sample)
                fout.write(json.dumps(sample) + "\n")

    import shutil; shutil.rmtree(temp_val_dir)
    print("Finished building both TRAIN and VAL datasets.")

def main():
    build_train_and_val_datasets()

if __name__ == "__main__":
    main()