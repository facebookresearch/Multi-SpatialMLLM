# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import random
import os
from tqdm import tqdm
import mmengine
import cv2
import numpy
# set seed
numpy.random.seed(6)
random.seed(6)

from spatial_engine.utils.scannet_utils.handler.info_handler import SceneInfoHandler, VisibilityInfoHandler

import argparse
import sys

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

class DepthComparisonDotQAEngine:
    def __init__(self, scene_info_path,
                 version_num="v1_0",
                 all_max_samples=-1,
                 image_output_dir=None,
                 visibility_info_path=None,
                 max_n_points_per_image=1,
                 warning_file=None):
        self.scene_info = SceneInfoHandler(scene_info_path)
        self.version_num = version_num
        self.image_output_dir = image_output_dir
        self.all_max_samples = all_max_samples
        self.task_name = "depth_comparison_dot"
        # * note, in v1_0, max_n_points_per_image is 1
        # * and even if it is set to be larger than 1, it will generate different QA pairs, i.e., single-round QA and the total number of samples is max_n_points_per_image * num_images (all_max_samples * max_n_points_per_image)
        self.max_n_points_per_image = max_n_points_per_image 
        self.warning_file = warning_file
        # read visibility infos
        self.visibility_info = VisibilityInfoHandler(visibility_info_path)
        assert max_n_points_per_image == 1, "If you want to set more points (pairs) per image, remember to check \
            if you want single-round QA or multi-round QAs. Currently support single-round QAs only."

        self.task_description = [
            "<image>\nGiven an image with two annotated points labeled with letters, determine which point is closer to or farther from the camera.",
            "<image>\nCompare the depths of two lettered points in the image and identify which one is nearer to or further from the viewer.",
            "<image>\nIn the provided image, two points are marked with letters. Your task is to find out which point is closer to the camera and which one is farther away.",
            "<image>\nExamine the image with two annotated points and decide which point is at a shorter distance from the camera and which one is at a greater distance.",
            "<image>\nGiven an image with two marked points, identify which point is positioned closer to the camera and which one is positioned farther from it.",
            "<image>\nAnalyze the image with two lettered points and determine which point is nearer to the camera and which one is farther from the camera.",
            "<image>\nIn the image provided, two points are labeled with letters. Determine the relative depth of these points and identify which one is closer to the camera.",
            "<image>\nLook at the image with two annotated points and decide which point is closer to the camera and which one is farther from the camera.",
            "<image>\nGiven an image with two marked points, determine which point is at a shorter distance from the camera and which one is at a greater distance.",
            "<image>\nCompare the depths of two lettered points in the image and identify which one is closer to the camera and which one is farther from the camera.",
            "<image>\nIn the provided image, two points are marked with letters. Your task is to find out which point is nearer to the camera and which one is farther away.",
            "<image>\nExamine the image with two annotated points and decide which point is positioned closer to the camera and which one is positioned farther from it.",
            "<image>\nGiven an image with two marked points, identify which point is at a shorter distance from the camera and which one is at a greater distance.",
            "<image>\nAnalyze the image with two lettered points and determine which point is closer to the camera and which one is farther from the camera.",
            "<image>\nIn the image provided, two points are labeled with letters. Determine the relative depth of these points and identify which one is nearer to the camera.",
            "<image>\nLook at the image with two annotated points and decide which point is closer to the camera and which one is farther from the camera.",
            "<image>\nGiven an image with two marked points, determine which point is at a shorter distance from the camera and which one is at a greater distance.",
            "<image>\nCompare the depths of two lettered points in the image and identify which one is closer to the camera and which one is farther from the camera.",
            "<image>\nIn the provided image, two points are marked with letters. Your task is to find out which point is nearer to the camera and which one is farther away.",
            "<image>\nExamine the image with two annotated points and decide which point is positioned closer to the camera and which one is positioned farther from it.",
            "<image>\nGiven an image with two marked points, identify which point is at a shorter distance from the camera and which one is at a greater distance.",
            "<image>\nAnalyze the image with two lettered points and determine which point is closer to the camera and which one is farther from the camera.",
            "<image>\nIn the image provided, two points are labeled with letters. Determine the relative depth of these points and identify which one is nearer to the camera.",
            "<image>\nLook at the image with two annotated points and decide which point is closer to the camera and which one is farther from the camera.",
            "<image>\nGiven an image with two marked points, determine which point is at a shorter distance from the camera and which one is at a greater distance.",
            "<image>\nCompare the depths of two lettered points in the image and identify which one is closer to the camera and which one is farther from the camera.",
            "<image>\nIn the provided image, two points are marked with letters. Your task is to find out which point is nearer to the camera and which one is farther away.",
            "<image>\nExamine the image with two annotated points and decide which point is positioned closer to the camera and which one is positioned farther from it.",
            "<image>\nGiven an image with two marked points, identify which point is at a shorter distance from the camera and which one is at a greater distance.",
            "<image>\nAnalyze the image with two lettered points and determine which point is closer to the camera and which one is farther from the camera."
        ]
        self.templates = {
            "closer_questions": [
                "Which of the annotated points is closer to the camera?",
                "Between the two annotated points, which one is nearer to the viewer?", 
                "Of the two marked points, which one is closer to the camera?",
                "Can you identify which annotated point has the shorter distance to the camera?",
                "Looking at the marked points, which appears closer to the viewing position?",
                "Which of the labeled points is positioned nearer to the camera?",
                "From the two annotated locations, which point is closer to the viewer?",
                "Among the marked points, which has less distance from the camera?",
                "Could you determine which annotated point is nearer to the viewing position?",
                "Which of the two labeled points appears to be closer to the camera?",
                "Between the marked locations, which point has the shorter distance?",
                "Can you tell which of the annotated points is nearest to the camera?",
                "Which of the two marked positions is closer to the viewing point?",
                "Looking at both annotations, which point is nearer to the camera?",
                "From the labeled points, which one has the smaller distance?",
                "Please identify the closer of the two annotated points.",
                "Which marked point would you say is nearer to the viewer?",
                "Of the annotated locations, which is closer to the camera position?",
                "Between these two marked points, which has less depth?",
                "Which annotation indicates the point closer to the camera?",
                "Which of the two points is positioned at a shorter distance from the camera?",
                "Can you determine which labeled point is closer to the camera?",
                "Which point, among the two, is closer to the camera?",
                "Identify the point that is closer to the camera from the two annotations.",
                "Which of the two points is nearest to the camera?",
                "Which point is positioned closer to the camera?",
                "Can you tell which point is closer to the camera?",
                "Which of the two points is at a shorter distance from the camera?",
                "Which point is closer to the camera among the two?",
                "Identify the closer point to the camera from the two annotations."
            ],
            "farther_questions": [
                "Which of the annotated points is farther from the camera?",
                "Between the two annotated points, which one is at a greater distance from the viewer?",
                "Of the two marked points, which one is more distant from the camera?",
                "Can you identify which annotated point has the longer distance to the camera?",
                "Looking at the marked points, which appears farther from the viewing position?",
                "Which of the labeled points is positioned more distant from the camera?",
                "From the two annotated locations, which point is farther from the viewer?",
                "Among the marked points, which has greater distance from the camera?",
                "Could you determine which annotated point is more distant from the viewing position?",
                "Which of the two labeled points appears to be farther from the camera?",
                "Between the marked locations, which point has the longer distance?",
                "Can you tell which of the annotated points is farthest from the camera?",
                "Which of the two marked positions is more distant from the viewing point?",
                "Looking at both annotations, which point is farther from the camera?",
                "From the labeled points, which one has the larger distance?",
                "Please identify the more distant of the two annotated points.",
                "Which marked point would you say is farther from the viewer?",
                "Of the annotated locations, which is more distant from the camera position?",
                "Between these two marked points, which has greater depth?",
                "Which annotation indicates the point farther from the camera?",
                "Which of the two points is positioned at a greater distance from the camera?",
                "Can you determine which labeled point is farther from the camera?",
                "Which point, among the two, is farther from the camera?",
                "Identify the point that is farther from the camera from the two annotations.",
                "Which of the two points is farthest from the camera?",
                "Which point is positioned farther from the camera?",
                "Can you tell which point is farther from the camera?",
                "Which of the two points is at a longer distance from the camera?",
                "Which point is farther from the camera among the two?",
                "Identify the farther point to the camera from the two annotations."
            ],
            "closer_answers": [
                "Point `{correct_label}` is closer to the camera.",
                "Point `{correct_label}` is nearer to the viewer.",
                "`{correct_label}` is the closer point to the camera.",
                "The point labeled `{correct_label}` has the shorter distance to the camera.",
                "Point `{correct_label}` is positioned nearer to the viewing position.",
                "`{correct_label}` is the closest of the marked points.",
                "The annotated point `{correct_label}` is nearer to the camera.",
                "Point `{correct_label}` has the smallest distance from the viewer.",
                "The marked point `{correct_label}` is closer to the camera.",
                "Among the annotated points, `{correct_label}` is nearest to the viewing position.",
                "The point identified as `{correct_label}` is closer.",
                "`{correct_label}` shows the nearer point location.",
                "The annotation `{correct_label}` marks the closer point.",
                "Point `{correct_label}` has less distance to the camera.",
                "The closer point is labeled as `{correct_label}`.",
                "`{correct_label}` indicates the nearer position.",
                "The point marked `{correct_label}` is at a shorter distance.",
                "From the camera's view, `{correct_label}` is closer.",
                "Point `{correct_label}` represents the nearer location.",
                "The annotation `{correct_label}` shows the closer point.",
                "The point `{correct_label}` is the one closer to the camera.",
                "Point `{correct_label}` is the nearest to the camera.",
                "`{correct_label}` is the point with the shortest distance to the camera.",
                "The point labeled `{correct_label}` is the closest to the camera.",
                "Point `{correct_label}` is the one positioned nearest to the camera.",
                "`{correct_label}` is the point closest to the camera.",
                "The annotated point `{correct_label}` is the one closer to the camera.",
                "Point `{correct_label}` is the one with the smallest distance to the camera.",
                "The marked point `{correct_label}` is the closest to the camera.",
                "Among the annotated points, `{correct_label}` is the one nearest to the camera."
            ],
            "farther_answers": [
                "Point `{correct_label}` is farther from the camera.",
                "Point `{correct_label}` is more distant from the viewer.",
                "`{correct_label}` is the farther point from the camera.",
                "The point labeled `{correct_label}` has the longer distance to the camera.",
                "Point `{correct_label}` is positioned more distant from the viewing position.",
                "`{correct_label}` is the most distant of the marked points.",
                "The annotated point `{correct_label}` is farther from the camera.",
                "Point `{correct_label}` has the greatest distance from the viewer.",
                "The marked point `{correct_label}` is more distant from the camera.",
                "Among the annotated points, `{correct_label}` is farthest from the viewing position.",
                "The point identified as `{correct_label}` is farther.",
                "`{correct_label}` shows the more distant point location.",
                "The annotation `{correct_label}` marks the farther point.",
                "Point `{correct_label}` has greater distance to the camera.",
                "The more distant point is labeled as `{correct_label}`.",
                "`{correct_label}` indicates the farther position.",
                "The point marked `{correct_label}` is at a longer distance.",
                "From the camera's view, `{correct_label}` is more distant.",
                "Point `{correct_label}` represents the farther location.",
                "The annotation `{correct_label}` shows the more distant point.",
                "The point `{correct_label}` is the one farther from the camera.",
                "Point `{correct_label}` is the farthest from the camera.",
                "`{correct_label}` is the point with the longest distance to the camera.",
                "The point labeled `{correct_label}` is the farthest from the camera.",
                "Point `{correct_label}` is the one positioned farthest from the camera.",
                "`{correct_label}` is the point farthest from the camera.",
                "The annotated point `{correct_label}` is the one farther from the camera.",
                "Point `{correct_label}` is the one with the greatest distance to the camera.",
                "The marked point `{correct_label}` is the farthest from the camera.",
                "Among the annotated points, `{correct_label}` is the one farthest from the camera."
            ],
        }

    def generate_random_point(self, height, width, margin=50):
        """Generate random point within image boundaries with margin"""
        x = random.randint(margin, width - margin)
        y = random.randint(margin, height - margin)
        return (x, y)

    def annotate_image(self, image, point):
        """Annotate image with a single point"""
        annotated_img = image.copy()
        x, y = point
        
        # Generate a random distinct color
        color = generate_distinct_colors(1)[0]
        
        # Draw circle
        cv2.circle(annotated_img, (x, y), 10, color, -1)
        
        return annotated_img

    def generate_qa_training_single_scene(self, scene_id):
        # Get all valid image IDs for the scene
        image_ids = self.scene_info.get_all_extrinsic_valid_image_ids(scene_id)
        scene_image_height, scene_image_width = self.scene_info.get_image_shape(scene_id)
        
        # Calculate how many images to sample from this scene
        if self.max_samples > 0:
            if self.max_samples > len(image_ids):
                n_images = self.max_samples
                sampled_image_ids = random.choices(image_ids, k=n_images)  # Sampling with replacement
            else:
                n_images = self.max_samples
                sampled_image_ids = random.sample(image_ids, n_images)  # Sampling without replacement
        else:
            n_images = len(image_ids)
            sampled_image_ids = random.sample(image_ids, n_images)  # Sampling without replacement, use all
        
        all_samples = []
        for image_id in sampled_image_ids:
            # * Load all the visible points in this image
            visible_points = self.visibility_info.get_image_to_points_info(scene_id, image_id) # [point_index, ...]

            for _ in range(self.max_n_points_per_image):
                # sample two points
                retry = 0
                while retry <= 10:
                    points_pair = random.sample(visible_points, 2)
                    points_info = []
                    for i, single_point in enumerate(points_pair):
                        point_2d, point_depth = self.scene_info.get_point_2d_coordinates_in_image(
                            scene_id, image_id, single_point, align=True, check_visible=True, return_depth=True
                        ) # input point_id is 0-indexed
                    
                        if len(point_2d) == 0:
                            # If the point is not visible in the image, print a warning and skip it
                            message = f"Warning: Point-Id {single_point} is not visible in image {image_id} in scene {scene_id}.\n"
                            print(message.strip())
                            with open(self.warning_file, 'a') as wf:
                                wf.write(message.strip())
                            continue

                        x = round((point_2d[0][0] / scene_image_width) * 1000)
                        y = round((point_2d[0][1] / scene_image_height) * 1000)
                        depth = round(point_depth[0] * 1000)

                        points_info.append({
                            'x': x, 'y': y, 'depth': depth,
                            'coords': (int(point_2d[0][0]), int(point_2d[0][1])),
                            'letter': chr(65 + i)  # A, B
                        })

                    if len(points_info) != 2 or points_info[0]['depth'] == points_info[1]['depth']:
                        if len(points_info) == 2:
                            message = f"Warning: Points {points_pair} in image {image_id} in scene {scene_id} have the same depth.\n Skip this pair."
                        else:
                            message = f"Warning: Cannot find two visible points in image {image_id} in scene {scene_id}.\n Skip this pair."
                        print(message.strip())
                        with open(self.warning_file, 'a') as wf:
                            wf.write(message.strip())
                        retry += 1
                        continue
                    
                    break
                
                if retry > 10:
                    # fail
                    message = f"Failed to find valid pair after 10 retries in image {image_id} in scene {scene_id}."
                    print(message.strip())
                    with open(self.warning_file, 'a') as wf:
                        wf.write(message.strip())
                    continue

                # Randomly assign A/B labels to points
                letters = ['A', 'B']
                random.shuffle(letters)
                points_info_shuffled = random.sample(points_info, 2)
                for i, point_info in enumerate(points_info_shuffled):
                    point_info['letter'] = letters[i]

                # Determine which point is closer/farther
                p1, p2 = points_info_shuffled  # Now using shuffled points
                closer_point = p1 if p1['depth'] <= p2['depth'] else p2
                farther_point = p2 if p1['depth'] <= p2['depth'] else p1

                # Randomly choose between closer or farther question
                is_closer_question = random.choice([True, False])
                templates = self.templates

                question_template = random.choice(templates["closer_questions" if is_closer_question else "farther_questions"])
                answer_template = random.choice(templates["closer_answers" if is_closer_question else "farther_answers"])
                task_description = random.choice(self.task_description)

                # Draw letters on image
                img_path = self.scene_info.get_image_path(scene_id, image_id)
                img = cv2.imread(img_path)
                for point_info in points_info_shuffled:  # Use shuffled points for drawing
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    cv2.circle(img, point_info['coords'], 10, color, -1)
                    cv2.putText(img, point_info['letter'], 
                              (point_info['coords'][0]+15, point_info['coords'][1]+15),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                # Save annotated image
                os.makedirs(os.path.join(self.image_output_dir, scene_id), exist_ok=True)
                annotated_img_path = os.path.join(self.image_output_dir, scene_id,
                                                f"{image_id}_p{points_pair[0]}_p{points_pair[1]}_annotated.jpg")
                cv2.imwrite(annotated_img_path, img)

                question = question_template
                answer = answer_template.format(
                    correct_label=closer_point['letter'] if is_closer_question else farther_point['letter']
                )

                gt_value = closer_point['letter'] if is_closer_question else farther_point['letter']
                question_type = "depth_comparison_annotated"

                conversation = [
                    {"from": "human", "value": f"{task_description}\n{question}"},
                    {"from": "gpt", "value": answer}
                ]

                # Complete the training sample for the current image
                training_sample = {
                    "id": f"{scene_id}_{image_id}_p{points_pair[0]}_p{points_pair[1]}",
                    "image": [f"{scene_id}/{image_id}_p{points_pair[0]}_p{points_pair[1]}_annotated.jpg"],
                    "conversations": conversation,
                    "height_list": [scene_image_height],
                    "width_list": [scene_image_width],
                    "question_type": question_type,
                    "gt_value": gt_value,
                    "points_info": points_info_shuffled, # points_info contains x (norm coordinate), y, depth, letter, and ori_coordinates
                    "is_closer_question": is_closer_question
                }
                all_samples.append(training_sample)
            
        return all_samples

    def generate_qa_training_data(self, output_dir, save_file=True):
        scene_ids = self.scene_info.get_sorted_keys()

        # if max_samples is not -1, then we need to sample the scenes
        if self.all_max_samples > 0:
            # need to calculate how many samples for each scene
            self.max_samples = max(self.all_max_samples // len(scene_ids) + 1, 1)
            if self.max_samples == 1:
                scene_ids = random.sample(scene_ids, self.all_max_samples)
        else:
            self.max_samples = -1
        self.num_used_scenes = len(scene_ids)

        train_data = []
        for scene_id in tqdm(scene_ids, desc="Generating QA Training Data"):
            train_data.extend(self.generate_qa_training_single_scene(scene_id))
        
        if len(train_data) > self.all_max_samples:
            train_data = random.sample(train_data, self.all_max_samples)

        random.shuffle(train_data)

        if save_file:
            output_jsonl_filepath = f"{output_dir}/{self.task_name}.jsonl"

            mmengine.mkdir_or_exist(output_dir)
            with open(output_jsonl_filepath, 'w') as f:
                for entry in train_data:
                    f.write(json.dumps(entry) + '\n')
            print(f"[Train] Training data saved to {output_jsonl_filepath}. Generated {len(train_data)} samples in total.")
        else:
            return train_data
        
    def convert_train_sample_to_eval_sample(self, train_sample):
        conversation = train_sample["conversations"]
        train_sample["text"] = conversation[0]["value"]
        return train_sample

    def generate_qa_eval_data(self, output_dir):
        assert self.max_n_points_per_image == 1, "max_n_points_per_image should be 1 for evaluation"
        train_data = self.generate_qa_training_data(output_dir, save_file=False)
        all_data = [self.convert_train_sample_to_eval_sample(train_sample) for train_sample in train_data]

        output_jsonl_filepath = f"{output_dir}/{self.task_name}.jsonl"

        mmengine.mkdir_or_exist(output_dir)
        with open(output_jsonl_filepath, 'w') as f:
            for entry in all_data:
                f.write(json.dumps(entry) + '\n')

        print(f"[Eval] Evaluation data saved to {output_jsonl_filepath}. Generated {len(all_data)} samples in total.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_suffix", type=str, default="")

    parser.add_argument("--train_scene_info_path", type=str, 
                        default=f"data/scannet/scannet_instance_data/scenes_train_info_i_D5.pkl")
    parser.add_argument("--val_scene_info_path", type=str, 
                        default=f"data/scannet/scannet_instance_data/scenes_val_info_i_D5.pkl")
    parser.add_argument("--train_all_max_samples", type=int, default=500000)
    parser.add_argument("--val_all_max_samples", type=int, default=300)

    parser.add_argument("--output_dir_train", type=str, 
                        default=f"training_data/depth_comparison_dot")
    parser.add_argument("--output_dir_val", type=str, 
                        default=f"evaluation_data/depth_comparison_dot")

    parser.add_argument("--version_num", type=str, default="v1_0")
    args = parser.parse_args()

    args.output_dir_train = os.path.join(args.output_dir_train, args.version_num)
    args.output_dir_val = os.path.join(args.output_dir_val, args.version_num)
    args.image_output_dir_train = os.path.join(args.output_dir_train, "images")
    args.image_output_dir_val = os.path.join(args.output_dir_val, "images")

    mmengine.mkdir_or_exist(args.output_dir_train)
    mmengine.mkdir_or_exist(args.output_dir_val)

    # read the visibility info files
    # visibility info 文件（整体 pickle 文件，包含所有 scene 的信息）
    train_visibility_info_path = f"data/scannet/scannet_instance_data/train_visibility_info_D5.parquet"
    val_visibility_info_path = f"data/scannet/scannet_instance_data/val_visibility_info_D5.parquet" 

    val_warning_file = f"{args.output_dir_val}/val_warning.txt"
    train_warning_file = f"{args.output_dir_train}/train_warning.txt"

    print("Generating evaluation data...") # cost 3s to generate
    qa_engine_eval = DepthComparisonDotQAEngine(
        scene_info_path=args.val_scene_info_path,
        version_num=args.version_num,
        all_max_samples=args.val_all_max_samples,
        image_output_dir=args.image_output_dir_val,
        visibility_info_path=val_visibility_info_path,
        warning_file=val_warning_file
    )
    qa_engine_eval.generate_qa_eval_data(args.output_dir_val)

    print("Generating training data...") # cost 1.5 hours to generate, will generate 337523 samples
    qa_engine_train = DepthComparisonDotQAEngine(
        scene_info_path=args.train_scene_info_path,
        version_num=args.version_num,
        all_max_samples=args.train_all_max_samples,
        image_output_dir=args.image_output_dir_train,
        visibility_info_path=train_visibility_info_path,
        warning_file=train_warning_file
    )
    qa_engine_train.generate_qa_training_data(args.output_dir_train)