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
numpy.random.seed(5)
random.seed(5)

from spatial_engine.utils.scannet_utils.handler.info_handler import SceneInfoHandler, VisibilityInfoHandler

import argparse

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

class DepthEstimationDotQAEngine:
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
        self.task_name = "depth_estimation_dot"
        # * note, in v1_0, max_n_points_per_image is 1
        # * and even if it is set to be larger than 1, it will generate different QA pairs, i.e., single-round QA and the total number of samples is max_n_points_per_image * num_images (all_max_samples * max_n_points_per_image)
        self.max_n_points_per_image = max_n_points_per_image 
        self.warning_file = warning_file
        # read visibility infos
        self.visibility_info = VisibilityInfoHandler(visibility_info_path)

        self.task_description = [
            "<image>\nGiven an image with an annotated point, complete the question-answer task.",
            "<image>\nFor an image with an annotated point, answer the depth-related questions.",
            "<image>\nUsing the provided image with an annotated point, complete the QA task.",
            "<image>\nGiven an image with a specific annotated point, perform the question-answer process.",
            "<image>\nWork with the image that has an annotated point to answer the related questions.",
            "<image>\nAnalyze the image with an annotated point and complete the QA task.",
            "<image>\nGiven an image where a point is annotated, proceed with the question-answer task.",
            "<image>\nFor the image with an annotated point, determine the depth-related answers.",
            "<image>\nUsing the image with an annotated point, perform the QA task.",
            "<image>\nGiven an image with a marked point, complete the question-answer process.",
            "<image>\nWork with the image containing an annotated point to answer the questions.",
            "<image>\nGiven an image with a highlighted point, complete the QA task.",
            "<image>\nFor an image with a marked point, answer the depth-related questions.",
            "<image>\nUsing the image with a highlighted point, perform the QA task.",
            "<image>\nGiven an image with a designated point, complete the question-answer process.",
            "<image>\nWork with the image that has a marked point to answer the related questions.",
            "<image>\nAnalyze the image with a designated point and complete the QA task.",
            "<image>\nGiven an image where a point is highlighted, proceed with the question-answer task.",
            "<image>\nFor the image with a designated point, determine the depth-related answers.",
            "<image>\nUsing the image with a marked point, perform the QA task.",
            "<image>\nGiven an image with a pinpointed location, engage in the question-answer task.",
            "<image>\nFor an image with a specified point, resolve the depth-related queries.",
            "<image>\nUtilize the image with a pinpointed spot to complete the QA task.",
            "<image>\nGiven an image with a noted point, carry out the question-answer process.",
            "<image>\nWork with the image that has a pinpointed point to answer the related questions.",
            "<image>\nExamine the image with a noted point and complete the QA task.",
            "<image>\nGiven an image where a point is pinpointed, proceed with the question-answer task.",
            "<image>\nFor the image with a noted point, ascertain the depth-related answers.",
            "<image>\nUsing the image with a pinpointed point, perform the QA task.",
            "<image>\nGiven an image with a specified point, complete the question-answer process."
        ]

        self.templates = {
            "questions": [
                "What is the depth of the annotated point in the image (in mm)?",
                "How far is the annotated point from the camera in millimeters?",
                "Determine the depth value of the annotated point in the given image (mm).",
                "Find the distance from the observer to the annotated point in the image, in mm.",
                "What is the measured depth of the annotated point in mm?",
                "How far away is the annotated point from the viewer in the image (mm)?",
                "Identify the depth value for the annotated point in millimeters.",
                "Given the annotated point, what is its depth in the image (mm)?",
                "What is the distance between the camera and the annotated point in mm?",
                "How deep is the annotated point in the given image (in millimeters)?",
                "What is the distance to the annotated point in the image (in mm)?",
                "How far is the annotated point located from the camera in mm?",
                "Determine the distance of the annotated point from the observer in mm.",
                "What is the depth measurement of the annotated point in the image (mm)?",
                "How far is the annotated point from the camera lens in millimeters?",
                "What is the depth of the highlighted point in the image (in mm)?",
                "How far is the highlighted point from the camera in millimeters?",
                "Determine the depth value of the highlighted point in the given image (mm).",
                "Find the distance from the observer to the highlighted point in the image, in mm.",
                "What is the measured depth of the highlighted point in mm?",
                "How far away is the highlighted point from the viewer in the image (mm)?",
                "Identify the depth value for the highlighted point in millimeters.",
                "Given the highlighted point, what is its depth in the image (mm)?",
                "What is the distance between the camera and the highlighted point in mm?",
                "How deep is the highlighted point in the given image (in millimeters)?",
                "What is the distance to the highlighted point in the image (in mm)?",
                "How far is the highlighted point located from the camera in mm?",
                "Determine the distance of the highlighted point from the observer in mm.",
                "What is the depth measurement of the highlighted point in the image (mm)?",
                "How far is the highlighted point from the camera lens in millimeters?"
            ],

            "answers": [
                "The depth of the annotated point is `{depth}` mm.",
                "It is `{depth}` mm away from the camera.",
                "The depth value of the annotated point is `{depth}` mm.",
                "The distance to the annotated point is `{depth}` mm.",
                "Measured depth of the annotated point is `{depth}` mm.",
                "The annotated point is located `{depth}` mm away from the viewer.",
                "The depth of the annotated point is `{depth}` mm.",
                "Its depth in the image is `{depth}` mm.",
                "The distance from the camera to the annotated point is `{depth}` mm.",
                "The annotated point is `{depth}` mm deep in the image.",
                "The depth of the highlighted point is `{depth}` mm.",
                "It is `{depth}` mm away from the camera.",
                "The depth value of the highlighted point is `{depth}` mm.",
                "The distance to the highlighted point is `{depth}` mm.",
                "Measured depth of the highlighted point is `{depth}` mm.",
                "The highlighted point is located `{depth}` mm away from the viewer.",
                "The depth of the highlighted point is `{depth}` mm.",
                "Its depth in the image is `{depth}` mm.",
                "The distance from the camera to the highlighted point is `{depth}` mm.",
                "The highlighted point is `{depth}` mm deep in the image.",
                "The depth of the marked point is `{depth}` mm.",
                "It is `{depth}` mm away from the camera.",
                "The depth value of the marked point is `{depth}` mm.",
                "The distance to the marked point is `{depth}` mm.",
                "Measured depth of the marked point is `{depth}` mm.",
                "The marked point is located `{depth}` mm away from the viewer.",
                "The depth of the marked point is `{depth}` mm.",
                "Its depth in the image is `{depth}` mm.",
                "The distance from the camera to the marked point is `{depth}` mm.",
                "The marked point is `{depth}` mm deep in the image."
            ]
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
            n_images = min(self.max_samples, len(image_ids))
        else:
            n_images = len(image_ids)
            
        # Randomly sample images
        sampled_image_ids = random.sample(image_ids, n_images)
        
        all_samples = []
        for image_id in sampled_image_ids:
            # * Load all the visible points in this image
            visible_points = self.visibility_info.get_image_to_points_info(scene_id, image_id) # [point_index, ...]

            # * Randomly sample a set of points from the visible points in the image
            # * need to consider the number of available points, if not enough, use put back sampling
            if len(visible_points) < self.max_n_points_per_image:
                sampled_points = random.choices(visible_points, k=self.max_n_points_per_image)
            else:
                sampled_points = random.sample(visible_points, self.max_n_points_per_image)

            for point in sampled_points:
                # Calculate the 2D coordinates of the point in the image
                point_2d, point_depth = self.scene_info.get_point_2d_coordinates_in_image(
                    scene_id, image_id, point, align=True, check_visible=True, return_depth=True
                )

                if len(point_2d) == 0:
                    # If the point is not visible in the image, print a warning and skip it
                    message = f"Warning: Point-Id {point} is not visible in image {image_id} in scene {scene_id}.\n"
                    print(message.strip())
                    with open(self.warning_file, 'a') as wf:
                        wf.write(message.strip())
                    continue

                # Convert the normalized coordinates to scaled values (0-1000 range)
                x = round((point_2d[0][0] / scene_image_width) * 1000)
                y = round((point_2d[0][1] / scene_image_height) * 1000)
                depth = round(point_depth[0] * 1000)  # depth is in meters

                # read and annotate the image
                img_path = self.scene_info.get_image_path(scene_id, image_id)
                img = cv2.imread(img_path)
                annotated_img = self.annotate_image(img, (int(point_2d[0][0]), int(point_2d[0][1])))

                # save the annotated image
                save_dir = os.path.join(self.image_output_dir, scene_id)
                mmengine.mkdir_or_exist(save_dir)
                save_path = os.path.join(save_dir, f"{image_id}_p{point}_annotated.jpg")
                cv2.imwrite(save_path, annotated_img)

                # Fill the question and answer templates with the coordinates and depth
                question_template = random.choice(self.templates["questions"])
                question = question_template

                answer_template = random.choice(self.templates["answers"])
                answer = answer_template.format(x1=x, y1=y, depth=depth)

                task_description = random.choice(self.task_description)

                # If it's the first round, add the task description
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

                # Complete the training sample for the current image
                training_sample = {
                    "id": f"{scene_id}_{image_id}_point{point}",
                    "image": [f"{scene_id}/{image_id}_p{point}_annotated.jpg"],
                    "conversations": conversation,
                    "height_list": [scene_image_height],
                    "width_list": [scene_image_width],
                    "question_type": "depth_estimation_dot",
                    "gt_value": depth,
                    "ori_coordinates": [int(point_2d[0][0]), int(point_2d[0][1])],
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
                        default=f"training_data/depth_estimation_dot")
    parser.add_argument("--output_dir_val", type=str, 
                        default=f"evaluation_data/depth_estimation_dot")

    parser.add_argument("--version_num", type=str, default="v1_0")
    args = parser.parse_args()

    args.output_dir_train = os.path.join(args.output_dir_train, args.version_num, args.output_suffix.replace('_', ''))
    args.output_dir_val = os.path.join(args.output_dir_val, args.version_num, args.output_suffix.replace('_', ''))
    args.image_output_dir_train = os.path.join(args.output_dir_train, "images")
    args.image_output_dir_val = os.path.join(args.output_dir_val, "images")

    # read the visibility info files
    train_visibility_info_path = f"data/scannet/scannet_instance_data/train_visibility_info_D5.parquet"
    val_visibility_info_path = f"data/scannet/scannet_instance_data/val_visibility_info_D5.parquet" 

    val_warning_file = f"{args.output_dir_val}/val_warning.txt"
    train_warning_file = f"{args.output_dir_train}/train_warning.txt"

    print("Generating evaluation data...") # cost 3s to generate
    qa_engine_eval = DepthEstimationDotQAEngine(
        scene_info_path=args.val_scene_info_path,
        version_num=args.version_num,
        all_max_samples=args.val_all_max_samples,
        image_output_dir=args.image_output_dir_val,
        visibility_info_path=val_visibility_info_path,
        warning_file=val_warning_file
    )
    qa_engine_eval.generate_qa_eval_data(args.output_dir_val)

    print("Generating training data...") # cost 1.5 hours to generate, will generate 337523 samples
    qa_engine_train = DepthEstimationDotQAEngine(
        scene_info_path=args.train_scene_info_path,
        version_num=args.version_num,
        all_max_samples=args.train_all_max_samples,
        image_output_dir=args.image_output_dir_train,
        visibility_info_path=train_visibility_info_path,
        warning_file=train_warning_file
    )
    qa_engine_train.generate_qa_training_data(args.output_dir_train)