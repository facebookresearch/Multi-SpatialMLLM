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
numpy.random.seed(7)
random.seed(7)

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

class DepthComparisonCoorQAEngine:
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
        self.task_name = "depth_comparison_coor"
        # * note, in v1_0, max_n_points_per_image is 1
        # * and even if it is set to be larger than 1, it will generate different QA pairs, i.e., single-round QA and the total number of samples is max_n_points_per_image * num_images (all_max_samples * max_n_points_per_image)
        self.max_n_points_per_image = max_n_points_per_image 
        self.warning_file = warning_file
        # read visibility infos
        self.visibility_info = VisibilityInfoHandler(visibility_info_path)
        assert max_n_points_per_image == 1, "If you want to set more points (pairs) per image, remember to check \
            if you want single-round QA or multi-round QAs. Currently support single-round QAs only."

        self.task_description = [
            "<image>\nGiven an image with two points specified by their coordinates, determine which point is closer to or farther from the camera. The coordinates [ x , y ] are normalized to 0-1 and scaled by 1000, with [ 0 , 0 ] at the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nCompare the depths of two points in the image specified by their coordinates. The coordinates [ x , y ] are normalized from 0 to 1 and multiplied by 1000, starting from the top-left corner [ 0 , 0 ]. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nIdentify which of the two points given by their coordinates is closer to or farther from the camera. The coordinates [ x , y ] are normalized between 0 and 1 and scaled by 1000, with [ 0 , 0 ] at the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nDetermine the relative depth of two points in the image based on their coordinates. The coordinates [ x , y ] are normalized from 0 to 1 and scaled by 1000, starting from the top-left corner [ 0 , 0 ]. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nGiven two points in the image, find out which one is nearer to or farther from the camera. The coordinates [ x , y ] are normalized to a range of 0-1 and scaled by 1000, with [ 0 , 0 ] at the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nAssess the depth of two points in the image using their coordinates. The coordinates [ x , y ] are normalized between 0 and 1 and multiplied by 1000, starting from the top-left corner [ 0 , 0 ]. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nEvaluate which of the two points specified by their coordinates is closer to or farther from the camera. The coordinates [ x , y ] are normalized to 0-1 and scaled by 1000, with [ 0 , 0 ] at the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nCompare the distances of two points in the image from the camera using their coordinates. The coordinates [ x , y ] are normalized from 0 to 1 and scaled by 1000, starting from the top-left corner [ 0 , 0 ]. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nGiven an image with two points, determine which one is closer to or farther from the camera based on their coordinates. The coordinates [ x , y ] are normalized to a range of 0-1 and scaled by 1000, with [ 0 , 0 ] at the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nAnalyze the depth of two points in the image using their coordinates. The coordinates [ x , y ] are normalized between 0 and 1 and multiplied by 1000, starting from the top-left corner [ 0 , 0 ]. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nIdentify which of the two points given by their coordinates is farther from or closer to the camera. The coordinates [ x , y ] are normalized between 0 and 1 and scaled by 1000, with [ 0 , 0 ] at the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nDetermine the relative distance of two points in the image from the camera based on their coordinates. The coordinates [ x , y ] are normalized from 0 to 1 and scaled by 1000, starting from the top-left corner [ 0 , 0 ]. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nGiven two points in the image, find out which one is farther from or closer to the camera. The coordinates [ x , y ] are normalized to a range of 0-1 and scaled by 1000, with [ 0 , 0 ] at the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nCompare the depths of two points in the image specified by their coordinates. The coordinates [ x , y ] are normalized from 0 to 1 and multiplied by 1000, starting from the top-left corner [ 0 , 0 ]. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nEvaluate the proximity of two points to the camera using their coordinates. The coordinates [ x , y ] are normalized between 0 and 1 and scaled by 1000, with [ 0 , 0 ] at the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nDetermine which of the two points is at a greater distance from or closer to the camera based on their coordinates. The coordinates [ x , y ] are normalized from 0 to 1 and scaled by 1000, starting from the top-left corner [ 0 , 0 ]. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nIdentify the point that is closer to or farther from the camera from the given coordinates. The coordinates [ x , y ] are normalized to 0-1 and scaled by 1000, with [ 0 , 0 ] at the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nAnalyze which point is farther from or closer to the camera using the given coordinates. The coordinates [ x , y ] are normalized between 0 and 1 and multiplied by 1000, starting from the top-left corner [ 0 , 0 ]. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nDetermine the closer point to or farther from the camera from the two given coordinates. The coordinates [ x , y ] are normalized from 0 to 1 and scaled by 1000, with [ 0 , 0 ] at the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nEvaluate which point is farther from or closer to the camera based on the coordinates provided. The coordinates [ x , y ] are normalized to a range of 0-1 and scaled by 1000, starting from the top-left corner [ 0 , 0 ]. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nIdentify the point with the lesser or greater depth from the camera using the coordinates. The coordinates [ x , y ] are normalized between 0 and 1 and scaled by 1000, with [ 0 , 0 ] at the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nDetermine which point is more distant from or closer to the camera using the given coordinates. The coordinates [ x , y ] are normalized from 0 to 1 and multiplied by 1000, starting from the top-left corner [ 0 , 0 ]. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nAnalyze the coordinates to find out which point is closer to or farther from the camera. The coordinates [ x , y ] are normalized to 0-1 and scaled by 1000, with [ 0 , 0 ] at the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nEvaluate the coordinates to determine which point is farther from or closer to the camera. The coordinates [ x , y ] are normalized between 0 and 1 and scaled by 1000, starting from the top-left corner [ 0 , 0 ]. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nIdentify the point that is nearest to or farther from the camera using the given coordinates. The coordinates [ x , y ] are normalized from 0 to 1 and scaled by 1000, with [ 0 , 0 ] at the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nDetermine which point is at a greater distance from or closer to the camera using the coordinates. The coordinates [ x , y ] are normalized to a range of 0-1 and multiplied by 1000, starting from the top-left corner [ 0 , 0 ]. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nAnalyze the coordinates to find out which point is farther from or closer to the camera. The coordinates [ x , y ] are normalized between 0 and 1 and scaled by 1000, with [ 0 , 0 ] at the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nEvaluate the coordinates to determine which point is closer to or farther from the camera. The coordinates [ x , y ] are normalized from 0 to 1 and multiplied by 1000, starting from the top-left corner [ 0 , 0 ]. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nIdentify the point that is farther from or closer to the camera using the given coordinates. The coordinates [ x , y ] are normalized to 0-1 and scaled by 1000, with [ 0 , 0 ] at the top-left corner. The x-axis represents the width, and the y-axis represents the height.",
            "<image>\nCompare the depths of two points in the image to determine which one is closer to or farther from the camera. The coordinates [ x , y ] are normalized from 0 to 1 and scaled by 1000, starting from the top-left corner [ 0 , 0 ]. The x-axis represents the width, and the y-axis represents the height."
        ]

        self.templates = {
            "closer_questions": [
                "Which point is closer to the camera: [ {x1} , {y1} ] or [ {x2} , {y2} ]?",
                "Between points [ {x1} , {y1} ] and [ {x2} , {y2} ], which one is nearer to the viewer?", 
                "Of the two points [ {x1} , {y1} ] and [ {x2} , {y2} ], which one is closer to the camera?",
                "Looking at points [ {x1} , {y1} ] and [ {x2} , {y2} ], which is at a shorter distance from the camera?",
                "Can you identify which point - [ {x1} , {y1} ] or [ {x2} , {y2} ] - has less distance to the camera?",
                "Which of these coordinates is nearer: [ {x1} , {y1} ] or [ {x2} , {y2} ]?",
                "Between the coordinates [ {x1} , {y1} ] and [ {x2} , {y2} ], which point is positioned closer to the viewer?",
                "Which of the two locations [ {x1} , {y1} ] and [ {x2} , {y2} ] has a shorter distance to the camera?",
                "Comparing the points [ {x1} , {y1} ] and [ {x2} , {y2} ], which one is nearest to the camera position?",
                "From the camera's perspective, which point - [ {x1} , {y1} ] or [ {x2} , {y2} ] - is closer?",
                "Could you determine which coordinate - [ {x1} , {y1} ] or [ {x2} , {y2} ] - is nearer to the viewing position?",
                "Among the two points [ {x1} , {y1} ] and [ {x2} , {y2} ], which has the shorter distance to the camera?",
                "In terms of proximity to the camera, which point is closer: [ {x1} , {y1} ] or [ {x2} , {y2} ]?",
                "Which coordinate pair represents the closer point: [ {x1} , {y1} ] or [ {x2} , {y2} ]?",
                "Between these two locations [ {x1} , {y1} ] and [ {x2} , {y2} ], which is nearer to the camera's position?",
                "Can you tell which point has less depth: [ {x1} , {y1} ] or [ {x2} , {y2} ]?",
                "Looking at the scene, which coordinate - [ {x1} , {y1} ] or [ {x2} , {y2} ] - is closer to the viewer?",
                "Which of these two points [ {x1} , {y1} ] and [ {x2} , {y2} ] has the shorter distance from the camera?",
                "From these coordinates, which is nearer to the camera: [ {x1} , {y1} ] or [ {x2} , {y2} ]?",
                "Please identify which point - [ {x1} , {y1} ] or [ {x2} , {y2} ] - is closer to the viewing position?",
                "Which point is at a shorter distance from the camera: [ {x1} , {y1} ] or [ {x2} , {y2} ]?",
                "Between the points [ {x1} , {y1} ] and [ {x2} , {y2} ], which one is closer to the camera?",
                "Which point is closer to the camera: [ {x1} , {y1} ] or [ {x2} , {y2} ]?",
                "Which of the two points [ {x1} , {y1} ] and [ {x2} , {y2} ] is closer to the camera?",
                "Which point is nearer to the camera: [ {x1} , {y1} ] or [ {x2} , {y2} ]?",
                "Which point is closer to the camera: [ {x1} , {y1} ] or [ {x2} , {y2} ]?",
                "Which of the two points [ {x1} , {y1} ] and [ {x2} , {y2} ] is closer to the camera?",
                "Which point is nearer to the camera: [ {x1} , {y1} ] or [ {x2} , {y2} ]?",
                "Which point is closer to the camera: [ {x1} , {y1} ] or [ {x2} , {y2} ]?",
                "Which of the two points [ {x1} , {y1} ] and [ {x2} , {y2} ] is closer to the camera?"
            ],
            "farther_questions": [
                "Which point is farther from the camera: [ {x1} , {y1} ] or [ {x2} , {y2} ]?",
                "Between points [ {x1} , {y1} ] and [ {x2} , {y2} ], which one is at a greater distance from the viewer?",
                "Of the two points [ {x1} , {y1} ] and [ {x2} , {y2} ], which one is more distant from the camera?",
                "Looking at points [ {x1} , {y1} ] and [ {x2} , {y2} ], which is at a greater distance from the camera?",
                "Can you identify which point - [ {x1} , {y1} ] or [ {x2} , {y2} ] - has more distance to the camera?",
                "Which of these coordinates is more remote: [ {x1} , {y1} ] or [ {x2} , {y2} ]?",
                "Between the coordinates [ {x1} , {y1} ] and [ {x2} , {y2} ], which point is positioned farther from the viewer?",
                "Which of the two locations [ {x1} , {y1} ] and [ {x2} , {y2} ] has a longer distance to the camera?",
                "Comparing the points [ {x1} , {y1} ] and [ {x2} , {y2} ], which one is furthest from the camera position?",
                "From the camera's perspective, which point - [ {x1} , {y1} ] or [ {x2} , {y2} ] - is more distant?",
                "Could you determine which coordinate - [ {x1} , {y1} ] or [ {x2} , {y2} ] - is farther from the viewing position?",
                "Among the two points [ {x1} , {y1} ] and [ {x2} , {y2} ], which has the longer distance to the camera?",
                "In terms of distance from the camera, which point is farther: [ {x1} , {y1} ] or [ {x2} , {y2} ]?",
                "Which coordinate pair represents the more distant point: [ {x1} , {y1} ] or [ {x2} , {y2} ]?",
                "Between these two locations [ {x1} , {y1} ] and [ {x2} , {y2} ], which is more remote from the camera's position?",
                "Can you tell which point has greater depth: [ {x1} , {y1} ] or [ {x2} , {y2} ]?",
                "Looking at the scene, which coordinate - [ {x1} , {y1} ] or [ {x2} , {y2} ] - is more distant from the viewer?",
                "Which of these two points [ {x1} , {y1} ] and [ {x2} , {y2} ] has the greater distance from the camera?",
                "From these coordinates, which is farther from the camera: [ {x1} , {y1} ] or [ {x2} , {y2} ]?",
                "Please identify which point - [ {x1} , {y1} ] or [ {x2} , {y2} ] - is more distant from the viewing position?",
                "Which point is at a greater distance from the camera: [ {x1} , {y1} ] or [ {x2} , {y2} ]?",
                "Between the points [ {x1} , {y1} ] and [ {x2} , {y2} ], which one is farther from the camera?",
                "Which point is farther from the camera: [ {x1} , {y1} ] or [ {x2} , {y2} ]?",
                "Which of the two points [ {x1} , {y1} ] and [ {x2} , {y2} ] is farther from the camera?",
                "Which point is more distant from the camera: [ {x1} , {y1} ] or [ {x2} , {y2} ]?",
                "Which point is farther from the camera: [ {x1} , {y1} ] or [ {x2} , {y2} ]?",
                "Which of the two points [ {x1} , {y1} ] and [ {x2} , {y2} ] is farther from the camera?",
                "Which point is more distant from the camera: [ {x1} , {y1} ] or [ {x2} , {y2} ]?",
                "Which point is farther from the camera: [ {x1} , {y1} ] or [ {x2} , {y2} ]?",
                "Which of the two points [ {x1} , {y1} ] and [ {x2} , {y2} ] is farther from the camera?"
            ],
            "closer_answers": [
                "Point `[ {correct_x} , {correct_y} ]` is closer to the camera.",
                "The point at `[ {correct_x} , {correct_y} ]` is nearer to the viewer.",
                "`[ {correct_x} , {correct_y} ]` is the closer point to the camera.",
                "The coordinates `[ {correct_x} , {correct_y} ]` represent the closer point.",
                "The point located at `[ {correct_x} , {correct_y} ]` has the shorter distance to the camera.",
                "`[ {correct_x} , {correct_y} ]` is positioned nearest to the viewing position.",
                "The point with coordinates `[ {correct_x} , {correct_y} ]` is closer to the viewer.",
                "Among the two points, `[ {correct_x} , {correct_y} ]` is nearer to the camera.",
                "The closer point is at coordinates `[ {correct_x} , {correct_y} ]`.",
                "`[ {correct_x} , {correct_y} ]` has the shorter distance from the camera.",
                "The point at `[ {correct_x} , {correct_y} ]` is at a shorter distance from the viewer.",
                "Coordinates `[ {correct_x} , {correct_y} ]` mark the closer point to the camera.",
                "The nearest point is at `[ {correct_x} , {correct_y} ]`.",
                "`[ {correct_x} , {correct_y} ]` indicates the position of the closer point.",
                "The point with the shorter distance is at `[ {correct_x} , {correct_y} ]`.",
                "Looking at the scene, `[ {correct_x} , {correct_y} ]` is the closer point.",
                "The coordinates `[ {correct_x} , {correct_y} ]` show the nearer point.",
                "`[ {correct_x} , {correct_y} ]` represents the point with less depth.",
                "The point at `[ {correct_x} , {correct_y} ]` has the shorter distance to the camera.",
                "From the camera's view, `[ {correct_x} , {correct_y} ]` is the closer point.",
                "The point `[ {correct_x} , {correct_y} ]` is closer to the camera.",
                "The point at `[ {correct_x} , {correct_y} ]` is nearer to the viewer.",
                "`[ {correct_x} , {correct_y} ]` is the closer point to the camera.",
                "The coordinates `[ {correct_x} , {correct_y} ]` represent the closer point.",
                "The point located at `[ {correct_x} , {correct_y} ]` has the shorter distance to the camera.",
                "`[ {correct_x} , {correct_y} ]` is positioned nearest to the viewing position.",
                "The point with coordinates `[ {correct_x} , {correct_y} ]` is closer to the viewer.",
                "Among the two points, `[ {correct_x} , {correct_y} ]` is nearer to the camera.",
                "The closer point is at coordinates `[ {correct_x} , {correct_y} ]`.",
                "`[ {correct_x} , {correct_y} ]` has the shorter distance from the camera."
            ],
            "farther_answers": [
                "Point `[ {correct_x} , {correct_y} ]` is farther from the camera.",
                "The point at `[ {correct_x} , {correct_y} ]` is more distant from the viewer.",
                "`[ {correct_x} , {correct_y} ]` is the farther point from the camera.",
                "The coordinates `[ {correct_x} , {correct_y} ]` represent the more distant point.",
                "The point located at `[ {correct_x} , {correct_y} ]` has the greater distance to the camera.",
                "`[ {correct_x} , {correct_y} ]` is positioned farthest from the viewing position.",
                "The point with coordinates `[ {correct_x} , {correct_y} ]` is more distant from the viewer.",
                "Among the two points, `[ {correct_x} , {correct_y} ]` is farther from the camera.",
                "The more distant point is at coordinates `[ {correct_x} , {correct_y} ]`.",
                "`[ {correct_x} , {correct_y} ]` has the greater distance from the camera.",
                "The point at `[ {correct_x} , {correct_y} ]` is at a greater distance from the viewer.",
                "Coordinates `[ {correct_x} , {correct_y} ]` mark the farther point from the camera.",
                "The most distant point is at `[ {correct_x} , {correct_y} ]`.",
                "`[ {correct_x} , {correct_y} ]` indicates the position of the farther point.",
                "The point with the longer distance is at `[ {correct_x} , {correct_y} ]`.",
                "Looking at the scene, `[ {correct_x} , {correct_y} ]` is the more distant point.",
                "The coordinates `[ {correct_x} , {correct_y} ]` show the farther point.",
                "`[ {correct_x} , {correct_y} ]` represents the point with greater depth.",
                "The point at `[ {correct_x} , {correct_y} ]` has the longer distance to the camera.",
                "From the camera's view, `[ {correct_x} , {correct_y} ]` is the more distant point.",
                "The point `[ {correct_x} , {correct_y} ]` is farther from the camera.",
                "The point at `[ {correct_x} , {correct_y} ]` is more distant from the viewer.",
                "`[ {correct_x} , {correct_y} ]` is the farther point from the camera.",
                "The coordinates `[ {correct_x} , {correct_y} ]` represent the more distant point.",
                "The point located at `[ {correct_x} , {correct_y} ]` has the greater distance to the camera.",
                "`[ {correct_x} , {correct_y} ]` is positioned farthest from the viewing position.",
                "The point with coordinates `[ {correct_x} , {correct_y} ]` is more distant from the viewer.",
                "Among the two points, `[ {correct_x} , {correct_y} ]` is farther from the camera.",
                "The more distant point is at coordinates `[ {correct_x} , {correct_y} ]`.",
                "`[ {correct_x} , {correct_y} ]` has the larger distance from the camera."
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
                    message = f"Warning: Points {points_pair} in image {image_id} in scene {scene_id} have the same depth.\n Skip this pair."
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

                question = question_template.format(
                    x1=p1['x'], y1=p1['y'], x2=p2['x'], y2=p2['y']
                )
                answer = answer_template.format(
                    correct_x=closer_point['x'] if is_closer_question else farther_point['x'],
                    correct_y=closer_point['y'] if is_closer_question else farther_point['y']
                )

                gt_value = [closer_point['x'], closer_point['y']] if is_closer_question else [farther_point['x'], farther_point['y']]
                question_type = "depth_comparison_coordinate"

                conversation = [
                    {"from": "human", "value": f"{task_description}\n{question}"},
                    {"from": "gpt", "value": answer}
                ]

                # Complete the training sample for the current image
                training_sample = {
                    "id": f"{scene_id}_{image_id}_p{points_pair[0]}_p{points_pair[1]}",
                    "image": [f"{scene_id}/{image_id}.jpg"], # {image_id}_p{points_pair[0]}_p{points_pair[1]}_annotated.jpg for dot
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
    parser.add_argument("--train_scene_info_path", type=str, 
                        default=f"data/scannet/scannet_instance_data/scenes_train_info_i_D5.pkl")
    parser.add_argument("--val_scene_info_path", type=str, 
                        default=f"data/scannet/scannet_instance_data/scenes_val_info_i_D5.pkl")
    parser.add_argument("--train_all_max_samples", type=int, default=500000)
    parser.add_argument("--val_all_max_samples", type=int, default=300)

    parser.add_argument("--output_dir_train", type=str, 
                        default=f"training_data/depth_comparison_coor")
    parser.add_argument("--output_dir_val", type=str, 
                        default=f"evaluation_data/depth_comparison_coor")

    parser.add_argument("--version_num", type=str, default="v1_0")
    args = parser.parse_args()

    args.output_dir_train = os.path.join(args.output_dir_train, args.version_num)
    args.output_dir_val = os.path.join(args.output_dir_val, args.version_num)
    args.image_output_dir_train = os.path.join(args.output_dir_train, "images")
    args.image_output_dir_val = os.path.join(args.output_dir_val, "images")

    # read the visibility info files
    # visibility info 文件（整体 pickle 文件，包含所有 scene 的信息）
    train_visibility_info_path = f"data/scannet/scannet_instance_data/train_visibility_info_D5.parquet"
    val_visibility_info_path = f"data/scannet/scannet_instance_data/val_visibility_info_D5.parquet" 

    val_warning_file = f"{args.output_dir_val}/val_warning.txt"
    train_warning_file = f"{args.output_dir_train}/train_warning.txt"

    mmengine.mkdir_or_exist(args.output_dir_train)
    mmengine.mkdir_or_exist(args.output_dir_val)

    print("Generating evaluation data...") # cost 3s to generate
    qa_engine_eval = DepthComparisonCoorQAEngine(
        scene_info_path=args.val_scene_info_path,
        version_num=args.version_num,
        all_max_samples=args.val_all_max_samples,
        image_output_dir=args.image_output_dir_val,
        visibility_info_path=val_visibility_info_path,
        warning_file=val_warning_file
    )
    qa_engine_eval.generate_qa_eval_data(args.output_dir_val)

    print("Generating training data...") # cost 1.5 hours to generate, will generate 337523 samples
    qa_engine_train = DepthComparisonCoorQAEngine(
        scene_info_path=args.train_scene_info_path,
        version_num=args.version_num,
        all_max_samples=args.train_all_max_samples,
        image_output_dir=args.image_output_dir_train,
        visibility_info_path=train_visibility_info_path,
        warning_file=train_warning_file
    )
    qa_engine_train.generate_qa_training_data(args.output_dir_train)