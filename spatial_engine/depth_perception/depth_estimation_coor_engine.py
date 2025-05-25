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
numpy.random.seed(4)
random.seed(4)

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

class DepthEstimationCoorQAEngine:
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
        self.task_name = "depth_estimation_coor"
        # * note, in v1_0, max_n_points_per_image is 1
        # * and even if it is set to be larger than 1, it will generate different QA pairs, i.e., single-round QA and the total number of samples is max_n_points_per_image * num_images (all_max_samples * max_n_points_per_image)
        self.max_n_points_per_image = max_n_points_per_image 
        self.warning_file = warning_file
        # read visibility infos
        self.visibility_info = VisibilityInfoHandler(visibility_info_path)

        self.task_description = [
            "<image>\nGiven a single image and a 2D point's coordinates, complete the question-answer task. The point coordinates [ x , y ] are normalized to a range of 0-1 and scaled by 1000, with [ 0 , 0 ] at the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nFor a given image and a point specified by its 2D coordinates, answer the depth-related questions. The coordinates [ x , y ] are normalized from 0 to 1 and multiplied by 1000, starting from the top-left corner [ 0 , 0 ]. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nUsing the provided image and the given point's 2D coordinates, complete the QA task. The coordinates [ x , y ] are scaled by 1000 after being normalized to a range of 0-1, with the origin located at the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nGiven an image and a specific point's 2D coordinates, perform the question-answer process. The coordinates [ x , y ] are normalized to 0-1 and then scaled by 1000, with [ 0 , 0 ] starting at the top-left. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nWork with the image and a point specified by its 2D coordinates to answer the related questions. The coordinates [ x , y ] are normalized to a range of 0-1 and scaled by 1000, originating from the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nAnalyze the image and use the 2D coordinates of a point to complete the QA task. The coordinates [ x , y ] are normalized from 0-1 and scaled by 1000, with the origin at the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nFor the given image, use the 2D coordinates of a point to answer depth-related questions. The coordinates [ x , y ] are normalized to 0-1 and scaled by 1000, starting from the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nUsing the image and the specified point's 2D coordinates, complete the QA task. The coordinates [ x , y ] are scaled by 1000 after normalization to a range of 0-1, with the origin at the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nGiven an image and a point's 2D coordinates, perform the question-answer process. The coordinates [ x , y ] are normalized to 0-1 and scaled by 1000, with [ 0 , 0 ] at the top-left. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nWork with the image and a point specified by its 2D coordinates to answer the questions. The coordinates [ x , y ] are normalized to a range of 0-1 and scaled by 1000, originating from the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nGiven a single image and a 2D point's coordinates, complete the QA task. The point coordinates [ x , y ] are normalized to a range of 0-1 and scaled by 1000, with [ 0 , 0 ] at the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nFor a given image and a point specified by its 2D coordinates, answer the questions. The coordinates [ x , y ] are normalized from 0 to 1 and multiplied by 1000, starting from the top-left corner [ 0 , 0 ]. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nUsing the provided image and the given point's 2D coordinates, complete the task. The coordinates [ x , y ] are scaled by 1000 after being normalized to a range of 0-1, with the origin located at the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nGiven an image and a specific point's 2D coordinates, perform the QA process. The coordinates [ x , y ] are normalized to 0-1 and then scaled by 1000, with [ 0 , 0 ] starting at the top-left. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nWork with the image and a point specified by its 2D coordinates to answer questions. The coordinates [ x , y ] are normalized to a range of 0-1 and scaled by 1000, originating from the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nAnalyze the image and use the 2D coordinates of a point to complete the task. The coordinates [ x , y ] are normalized from 0-1 and scaled by 1000, with the origin at the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nFor the given image, use the 2D coordinates of a point to answer questions. The coordinates [ x , y ] are normalized to 0-1 and scaled by 1000, starting from the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nUsing the image and the specified point's 2D coordinates, complete the task. The coordinates [ x , y ] are scaled by 1000 after normalization to a range of 0-1, with the origin at the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nGiven an image and a point's 2D coordinates, perform the process. The coordinates [ x , y ] are normalized to 0-1 and scaled by 1000, with [ 0 , 0 ] at the top-left. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nWork with the image and a point specified by its 2D coordinates to answer. The coordinates [ x , y ] are normalized to a range of 0-1 and scaled by 1000, originating from the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nGiven a single image and a 2D point's coordinates, complete the task. The point coordinates [ x , y ] are normalized to a range of 0-1 and scaled by 1000, with [ 0 , 0 ] at the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nFor a given image and a point specified by its 2D coordinates, answer. The coordinates [ x , y ] are normalized from 0 to 1 and multiplied by 1000, starting from the top-left corner [ 0 , 0 ]. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nUsing the provided image and the given point's 2D coordinates, complete. The coordinates [ x , y ] are scaled by 1000 after being normalized to a range of 0-1, with the origin located at the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nGiven an image and a specific point's 2D coordinates, perform. The coordinates [ x , y ] are normalized to 0-1 and then scaled by 1000, with [ 0 , 0 ] starting at the top-left. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nWork with the image and a point specified by its 2D coordinates to answer. The coordinates [ x , y ] are normalized to a range of 0-1 and scaled by 1000, originating from the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nAnalyze the image and use the 2D coordinates of a point to complete. The coordinates [ x , y ] are normalized from 0-1 and scaled by 1000, with the origin at the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nFor the given image, use the 2D coordinates of a point to answer. The coordinates [ x , y ] are normalized to 0-1 and scaled by 1000, starting from the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nUsing the image and the specified point's 2D coordinates, complete. The coordinates [ x , y ] are scaled by 1000 after normalization to a range of 0-1, with the origin at the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nGiven an image and a point's 2D coordinates, perform. The coordinates [ x , y ] are normalized to 0-1 and scaled by 1000, with [ 0 , 0 ] at the top-left. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nWork with the image and a point specified by its 2D coordinates to answer. The coordinates [ x , y ] are normalized to a range of 0-1 and scaled by 1000, originating from the top-left corner. The x-axis represents the width, and the y-axis represents the height."
        ]

        self.templates = {
            "questions": [
                "What is the depth at point [ {x1} , {y1} ] in the image (in mm)?",
                "How far is the point [ {x1} , {y1} ] from the camera in millimeters?",
                "Determine the depth value at the coordinates [ {x1} , {y1} ] in the given image (mm).",
                "Find the distance from the observer to the point [ {x1} , {y1} ] in the image, in mm.",
                "What is the measured depth of the point located at [ {x1} , {y1} ] in mm?",
                "How far away is the specified point [ {x1} , {y1} ] from the viewer in the image (mm)?",
                "Identify the depth value for the point at coordinates [ {x1} , {y1} ] in millimeters.",
                "Given the point [ {x1} , {y1} ], what is its depth in the image (in mm)?",
                "What is the distance between the camera and the point [ {x1} , {y1} ] in mm?",
                "How deep is the point at [ {x1} , {y1} ] in the given image (in millimeters)?",
                "What depth is recorded at the coordinates [ {x1} , {y1} ] in the image (in mm)?",
                "Can you measure the depth at point [ {x1} , {y1} ] in millimeters?",
                "What is the depth measurement for the point [ {x1} , {y1} ] in the image (in mm)?",
                "How much distance is there to the point [ {x1} , {y1} ] in mm?",
                "What is the depth at the specified coordinates [ {x1} , {y1} ] (in mm)?",
                "How far is the point [ {x1} , {y1} ] in the image from the camera (in mm)?",
                "What is the depth value at point [ {x1} , {y1} ] in millimeters?",
                "Can you find the depth at the coordinates [ {x1} , {y1} ] (in mm)?",
                "What is the distance to the point [ {x1} , {y1} ] in the image (in mm)?",
                "How deep is the point [ {x1} , {y1} ] in the image (in mm)?",
                "What is the depth at the given point [ {x1} , {y1} ] (in mm)?",
                "How far is the point [ {x1} , {y1} ] from the observer (in mm)?",
                "What is the depth at the coordinates [ {x1} , {y1} ] (in mm)?",
                "How far away is the point [ {x1} , {y1} ] in the image (in mm)?",
                "What is the depth at the point [ {x1} , {y1} ] (in mm)?",
                "How far is the point [ {x1} , {y1} ] in millimeters?",
                "What is the depth value at the point [ {x1} , {y1} ] (in mm)?",
                "How deep is the point [ {x1} , {y1} ] (in mm)?",
                "What is the depth at the coordinates [ {x1} , {y1} ] in the image (in mm)?",
                "How far is the point [ {x1} , {y1} ] from the camera in the image (in mm)?"
            ],

            "answers": [
                "The depth at point [ {x1} , {y1} ] is `{depth}` mm.",
                "It is `{depth}` mm away from the camera.",
                "The depth value at these coordinates is `{depth}` mm.",
                "The distance to the point is `{depth}` mm.",
                "Measured depth of this point is `{depth}` mm.",
                "The point is located `{depth}` mm away from the viewer.",
                "The depth at the given coordinates is `{depth}` mm.",
                "Its depth in the image is `{depth}` mm.",
                "The distance from the camera to this point is `{depth}` mm.",
                "The point is `{depth}` mm deep in the image.",
                "The depth at these coordinates is `{depth}` mm.",
                "It measures `{depth}` mm in depth.",
                "The depth recorded is `{depth}` mm.",
                "The point's depth is `{depth}` mm.",
                "Depth at this location is `{depth}` mm.",
                "The depth value is `{depth}` mm.",
                "This point is `{depth}` mm deep.",
                "The depth here is `{depth}` mm.",
                "It is `{depth}` mm in depth.",
                "The depth at this point is `{depth}` mm.",
                "The point [ {x1} , {y1} ] has a depth of `{depth}` mm.",
                "At point [ {x1} , {y1} ], the depth is `{depth}` mm.",
                "Point [ {x1} , {y1} ] is `{depth}` mm deep.",
                "Depth at point [ {x1} , {y1} ] is `{depth}` mm.",
                "The depth at point [ {x1} , {y1} ] measures `{depth}` mm.",
                "Point [ {x1} , {y1} ] has a depth value of `{depth}` mm.",
                "The depth at point [ {x1} , {y1} ] is recorded as `{depth}` mm.",
                "At point [ {x1} , {y1} ], depth is `{depth}` mm.",
                "The depth at point [ {x1} , {y1} ] is `{depth}` mm.",
                "Point [ {x1} , {y1} ] is measured at `{depth}` mm depth."
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

                # Fill the question and answer templates with the coordinates and depth
                question_template = random.choice(self.templates["questions"])
                question = question_template.format(x1=x, y1=y)

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
                    "image": [f"{scene_id}/{image_id}.jpg"],
                    "conversations": conversation,
                    "height_list": [scene_image_height],
                    "width_list": [scene_image_width],
                    "question_type": "depth_estimation_coor",
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
                        default=f"training_data/depth_estimation_coor")
    parser.add_argument("--output_dir_val", type=str, 
                        default=f"evaluation_data/depth_estimation_coor")

    parser.add_argument("--version_num", type=str, default="v1_0")
    args = parser.parse_args()

    args.output_dir_train = os.path.join(args.output_dir_train, args.version_num, args.output_suffix.replace('_', ''))
    args.output_dir_val = os.path.join(args.output_dir_val, args.version_num, args.output_suffix.replace('_', ''))
    args.image_output_dir_train = os.path.join(args.output_dir_train, "images")
    args.image_output_dir_val = os.path.join(args.output_dir_val, "images")

    # read the visibility info files
    # visibility info 文件（整体 pickle 文件，包含所有 scene 的信息）
    train_visibility_info_path = f"data/scannet/scannet_instance_data/train_visibility_info_D5.parquet"
    val_visibility_info_path = f"data/scannet/scannet_instance_data/val_visibility_info_D5.parquet" 

    val_warning_file = f"{args.output_dir_val}/val_warning.txt"
    train_warning_file = f"{args.output_dir_train}/train_warning.txt"

    print("Generating evaluation data...") # cost 3s to generate
    qa_engine_eval = DepthEstimationCoorQAEngine(
        scene_info_path=args.val_scene_info_path,
        version_num=args.version_num,
        all_max_samples=args.val_all_max_samples,
        image_output_dir=args.image_output_dir_val,
        visibility_info_path=val_visibility_info_path,
        warning_file=val_warning_file
    )
    qa_engine_eval.generate_qa_eval_data(args.output_dir_val)

    print("Generating training data...") # cost 51 mins to generate, will generate 331295 samples
    qa_engine_train = DepthEstimationCoorQAEngine(
        scene_info_path=args.train_scene_info_path,
        version_num=args.version_num,
        all_max_samples=args.train_all_max_samples,
        image_output_dir=args.image_output_dir_train,
        visibility_info_path=train_visibility_info_path,
        warning_file=train_warning_file
    )
    qa_engine_train.generate_qa_training_data(args.output_dir_train)