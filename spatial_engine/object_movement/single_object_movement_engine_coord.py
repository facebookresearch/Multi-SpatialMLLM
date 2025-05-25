# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Before running this script, you need to donwload all the npz files following the official script.
- Generally, we assume the structure of the data is like:
```
data/tapvid3d_dataset
├── adt
│   ├── "id".npz
├── pstudio
│   ├── "id".npz
```
"""

import random
from random import shuffle
random.seed(0)
import numpy as np
np.random.seed(0)
import cv2
import os
import mmengine

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from multiprocessing import Pool
import json

def smooth_distance_changes(distances_t, distances_prev_t, smoothing_factor=0.01):
    """
    Smooths the distance changes between two consecutive time steps by filtering out small changes.
    
    Parameters:
    distances_t (numpy.ndarray): Distance matrix at time t.
    distances_prev_t (numpy.ndarray): Distance matrix at time t-1.
    smoothing_factor (float): The threshold for smoothing; changes smaller than this value are ignored.
    
    Returns:
    numpy.ndarray: Smoothed distance change matrix.
    """
    distance_change = np.abs(distances_t - distances_prev_t)
    return np.where(distance_change > smoothing_factor, distance_change, 0)

def rigid_body_segmentation(points, threshold=0.1, smoothing_factor=0.01):
    """
    Segments points into rigid bodies based on smoothed distance changes over time.
    
    Parameters:
    points (numpy.ndarray): Time series data of the points, shaped as (T, N, 3), 
                            where T is the number of time steps, N is the number of points, 
                            and 3 represents the spatial coordinates.
    threshold (float): Distance threshold for hierarchical clustering, controlling sensitivity.
    smoothing_factor (float): Threshold for smoothing distance changes to reduce the effect of noise.
    
    Returns:
    list of list: Each sublist contains the indices of points that belong to the same rigid body.
    """
    T, N, _ = points.shape

    # Initialize cumulative loss matrix to accumulate distance changes between point pairs over time
    cumulative_loss = np.zeros((N, N))

    # Iterate over time steps to compute the cumulative distance changes between points
    for t in range(1, T):
        # Compute pairwise distances between points at time t and t-1
        distances_t = squareform(pdist(points[t]))  # Distance matrix at time t
        distances_prev_t = squareform(pdist(points[t - 1]))  # Distance matrix at time t-1
        
        # Smooth the distance changes to filter out small noise
        smoothed_change = smooth_distance_changes(distances_t, distances_prev_t, smoothing_factor)
        
        # Accumulate the smoothed distance changes
        cumulative_loss += smoothed_change

    # Perform hierarchical clustering using the average linkage method
    links = linkage(squareform(cumulative_loss), method='average')
    
    # Assign cluster labels based on the distance threshold
    labels = fcluster(links, threshold, criterion='distance')

    # Group points by their assigned cluster labels
    groups = []
    for i in range(1, max(labels) + 1):
        group = np.where(labels == i)[0].tolist()
        groups.append(group)
    
    return groups

def filter_large_groups(groups, min_size=5):
    """
    Filters the groups to retain only those with more than a specified number of points.
    
    Parameters:
    groups (list of list): The list of groups where each group contains indices of points.
    min_size (int): The minimum number of points a group should have to be retained.
    
    Returns:
    list of list: Filtered list of groups where each group has more than `min_size` points.
    """
    return [group for group in groups if len(group) > min_size]


TASK_DESCRIPTION = [
    "Image-1: <image>\nImage-2: <image>\nGiven two images, analyze the movements of objects in the images and the cameras that captured them. The movement should be relative to the first image. Note that the objects in the images and the camera may or may not have moved.",
    "Image-1: <image>\nImage-2: <image>\nCompare these two images and describe how objects and the camera have moved relative to their positions in the first image. Keep in mind that either or both may have changed position.",
    "Image-1: <image>\nImage-2: <image>\nExamine the spatial changes between these two images, considering both object and camera motion relative to the first frame. Note that movement could involve either or both.",
    "Image-1: <image>\nImage-2: <image>\nAnalyze how the scene has transformed between these two images, tracking both object and camera movements from their initial positions. Remember that either or neither might have moved.",
    "Image-1: <image>\nImage-2: <image>\nStudy these two images and identify any changes in position of objects or camera, using the first image as reference. Both elements may or may not show movement.",
    "Image-1: <image>\nImage-2: <image>\nObserve these two frames and detail any motion of objects or camera relative to the first image's configuration. Note that movement isn't guaranteed for either.",
    "Image-1: <image>\nImage-2: <image>\nEvaluate these two images to determine how objects and camera positioning have changed, using the first image as baseline. Be aware that motion may affect either, both, or neither.",
    "Image-1: <image>\nImage-2: <image>\nAssess the positional changes of both objects and camera between these two images, relative to the initial frame. Consider that movement isn't necessary for either element.",
    "Image-1: <image>\nImage-2: <image>\nReview these two images and describe any spatial changes in object positions or camera placement, using the first image as reference. Movement may involve either or both components.",
    "Image-1: <image>\nImage-2: <image>\nInvestigate the movement patterns in these two images, considering both object and camera motion relative to the first frame. Note that changes could affect either, both, or neither.",
    "Image-1: <image>\nImage-2: <image>\nDocument the spatial transitions between these two images, tracking both object and camera movement from their initial positions. Be aware that motion isn't guaranteed.",
    "Image-1: <image>\nImage-2: <image>\nIdentify any positional shifts in objects or camera placement between these two frames, using the first image as your reference point. Either or both may have moved.",
    "Image-1: <image>\nImage-2: <image>\nMap out the changes in object and camera positions between these two images, relative to the initial frame. Remember that movement isn't a requirement for either.",
    "Image-1: <image>\nImage-2: <image>\nTrack the spatial evolution from the first image to the second, considering both object and camera movement. Note that either, both, or neither may have changed position.",
    "Image-1: <image>\nImage-2: <image>\nDetail the positional changes between these two frames, examining both object and camera movement relative to the first image. Be aware that motion may affect either or both.",
    "Image-1: <image>\nImage-2: <image>\nCharacterize the spatial differences between these two images, focusing on object and camera movement from their initial positions. Note that change isn't mandatory for either.",
    "Image-1: <image>\nImage-2: <image>\nOutline any movement of objects or camera between these two frames, using the first image as your baseline. Consider that either or both may have shifted position.",
    "Image-1: <image>\nImage-2: <image>\nDescribe the spatial transformations visible in these two images, considering both object and camera motion relative to the first frame. Movement may involve either or both.",
    "Image-1: <image>\nImage-2: <image>\nExplore how the scene has changed between these two images, tracking object and camera movement from their initial positions. Note that either or neither might show motion.",
    "Image-1: <image>\nImage-2: <image>\nQuantify the positional changes between these two frames, examining both object and camera movement relative to the first image. Remember that motion isn't guaranteed.",
    "Image-1: <image>\nImage-2: <image>\nMeasure the spatial differences between these two images, focusing on both object and camera movement from their starting positions. Either or both may have moved.",
    "Image-1: <image>\nImage-2: <image>\nReport on any movement detected between these two frames, considering both object and camera motion relative to the first image. Note that changes could affect either or both.",
    "Image-1: <image>\nImage-2: <image>\nSummarize the positional changes visible in these two images, tracking both object and camera movement from their initial state. Be aware that motion isn't necessary.",
    "Image-1: <image>\nImage-2: <image>\nChart the spatial transitions between these two frames, examining both object and camera movement relative to the first image. Either, both, or neither may have moved.",
    "Image-1: <image>\nImage-2: <image>\nDocument how the scene has evolved between these two images, considering both object and camera motion from their starting positions. Movement could affect either or both.",
    "Image-1: <image>\nImage-2: <image>\nAnalyze the positional shifts between these two frames, tracking both object and camera movement relative to the first image. Note that either or neither might have moved.",
    "Image-1: <image>\nImage-2: <image>\nIdentify any spatial changes between these two images, examining both object and camera motion from their initial positions. Remember that movement isn't required.",
    "Image-1: <image>\nImage-2: <image>\nEvaluate the movement patterns between these two frames, considering both object and camera positioning relative to the first image. Either or both may show motion.",
    "Image-1: <image>\nImage-2: <image>\nAssess how the scene has transformed between these two images, tracking both object and camera movement from their starting points. Note that changes may affect either or both.",
    "Image-1: <image>\nImage-2: <image>\nReview the spatial differences between these two frames, examining both object and camera motion relative to the first image. Be aware that movement isn't guaranteed for either.",
    "Image-1: <image>\nImage-2: <image>\nMap the positional changes between these two images, considering both object and camera movement from their initial state. Either, both, or neither may have shifted.",
]

QUESTION_TEMPLATES = {
    "tapvid3d_total_distance": [
        "How far did the point at [ {x1} , {y1} ] in Image-1 travel between the two shots?",
        "What is the total distance the point at [ {x1} , {y1} ] in Image-1 moved from its first position?",
        "Could you give me the magnitude of the point's displacement from [ {x1} , {y1} ] in Image-1?",
        "I'm curious about the overall traveled distance of the point at [ {x1} , {y1} ] in Image-1 in millimeters?",
        "Please specify the total movement distance of the point at [ {x1} , {y1} ] in Image-1.",
        "From start to end, how many units (mm) did the point at [ {x1} , {y1} ] in Image-1 shift in space?",
        "What is the absolute distance covered by the point at [ {x1} , {y1} ] in Image-1 between these images?",
        "Please provide the length of the path the point at [ {x1} , {y1} ] in Image-1 took.",
        "Does the point at [ {x1} , {y1} ] in Image-1 have a large or small travel distance, and how much is it?",
        "Could you measure the total displacement for the point at [ {x1} , {y1} ] in Image-1's movement?",
        "Could you calculate the overall distance from the point at [ {x1} , {y1} ] in Image-1's first pose to the second?",
        "In numerical terms, how far did the point at [ {x1} , {y1} ] in Image-1 go?",
        "Is there a measurement for the point at [ {x1} , {y1} ] in Image-1's complete travel distance in mm?",
        "Between these two viewpoints, what's the movement distance of the point at [ {x1} , {y1} ] in Image-1?",
        "I want to know the total length of the point at [ {x1} , {y1} ] in Image-1's translation.",
        "Please share how many millimeters the point at [ {x1} , {y1} ] in Image-1 has moved in total.",
        "Is the point at [ {x1} , {y1} ] in Image-1's movement more than a few millimeters, and how many exactly?",
        "How do you quantify the complete distance traveled by the point at [ {x1} , {y1} ] in Image-1?",
        "What is the final measure of the distance the point at [ {x1} , {y1} ] in Image-1 traversed?",
        "How large is the gap between the point at [ {x1} , {y1} ] in Image-1's initial and final positions in mm?",
        "What's the point at [ {x1} , {y1} ] in Image-1's net travel distance from the first shot to the second shot?",
        "I'm interested in the point-to-point distance of the point at [ {x1} , {y1} ] in Image-1's shift.",
        "Could you give an estimate or exact number of how far the point at [ {x1} , {y1} ] in Image-1 has moved?",
        "Does the data say how many millimeters separate the point at [ {x1} , {y1} ] in Image-1's old and new positions?",
        "What's the measurement of the point at [ {x1} , {y1} ] in Image-1's movement vector's length?",
        "Check the final distance: how many mm did the point at [ {x1} , {y1} ] in Image-1 shift overall?",
        "In your analysis, what's the total movement distance for the point at [ {x1} , {y1} ] in Image-1?",
        "If we consider the start and end points, how far apart are they for the point at [ {x1} , {y1} ] in Image-1?",
        "Please clarify the total displacement distance in millimeters for the point at [ {x1} , {y1} ] in Image-1.",
        "Give me the final figure for the point at [ {x1} , {y1} ] in Image-1's traveled distance in mm?"
    ],
    "tapvid3d_displacement_vector": [
        "The first image is oriented such that positive X is right, Y is down, and Z is forward. What is the overall displacement vector of the point at [ {x1} , {y1} ] in Image-1?",
        "The first image is oriented such that positive X is right, Y is down, and Z is forward. Could you provide the point at [ {x1} , {y1} ] in Image-1's movement vector in 3D coordinates?",
        "The first image is oriented with X->right, Y->down, Z->forward. Please specify the `[ x , y , z ]` displacement vector of the point at [ {x1} , {y1} ] in Image-1 in mm?",
        "The first image sets X=right, Y=down, Z=forward as positive. I'm curious about the entire translation vector from the first to second position for the point at [ {x1} , {y1} ] in Image-1.",
        "Given X->right, Y->down, Z->forward, what does the point at [ {x1} , {y1} ] in Image-1's shift vector look like in millimeters for each axis?",
        "We define X as right, Y as down, Z as forward. How do you represent the 3D displacement of the point at [ {x1} , {y1} ] in Image-1 as a vector?",
        "Considering the first image orientation (X=right, Y=down, Z=forward), could you express the point at [ {x1} , {y1} ] in Image-1's movement as a coordinate vector in mm?",
        "The first image uses positive X=right, Y=down, Z=forward. From the start to the end, what's the `[ x , y , z ]` translation of the point at [ {x1} , {y1} ] in Image-1?",
        "With X=right, Y=down, Z=forward, I want the point at [ {x1} , {y1} ] in Image-1's movement in vector form, specifying the offsets in each axis.",
        "Remember X->right, Y->down, Z->forward. Please give me the exact displacement as `[ x , y , z ]` in mm for the point at [ {x1} , {y1} ] in Image-1.",
        "The coordinate system is X=right, Y=down, Z=forward. Do we have the final 3D translation vector for the point at [ {x1} , {y1} ] in Image-1's shift?",
        "We assume positive X=right, Y=down, Z=forward. How can we break down the point at [ {x1} , {y1} ] in Image-1's movement into x, y, z components in mm?",
        "The first image orientation sets X, Y, Z as right, down, forward. Which vector best describes the point at [ {x1} , {y1} ] in Image-1's overall translation?",
        "Using X=right, Y=down, Z=forward, could you detail the point at [ {x1} , {y1} ] in Image-1's movement vector so I see how it moved along each axis?",
        "Positive X is right, Y is down, Z is forward. Is there a coordinate triple representing the point at [ {x1} , {y1} ] in Image-1's new position minus the old?",
        "If we consider X=right, Y=down, Z=forward, tell me the displacement vector that sums up how the point at [ {x1} , {y1} ] in Image-1 traveled in `[ x , y , z ]` mm.",
        "The coordinate definition is X->right, Y->down, Z->forward. What's the resulting 3D vector for the point at [ {x1} , {y1} ] in Image-1's net translation?",
        "In a system where X=right, Y=down, Z=forward, I'd like the translation vector: `[ x_movement , y_movement , z_movement ]` in mm for the point at [ {x1} , {y1} ] in Image-1.",
        "Given the orientation X=right, Y=down, Z=forward, does the data give us the point at [ {x1} , {y1} ] in Image-1's movement as a standard `[ x , y , z ]` vector?",
        "From the start to end, with X=right, Y=down, Z=forward, what's the `[ x , y , z ]` vector describing the point at [ {x1} , {y1} ] in Image-1's translation?",
        "Under the assumption X=right, Y=down, Z=forward, I want the final movement as a three-dimensional vector describing the point at [ {x1} , {y1} ] in Image-1's translation.",
        "Taking X=right, Y=down, Z=forward as the scene basis, how do we express the total movement as `[ x , y , z ]` in mm for the point at [ {x1} , {y1} ] in Image-1?",
        "If the first image orientation is X=right, Y=down, Z=forward, could you share the numeric triple for the point at [ {x1} , {y1} ] in Image-1's displacement along `[ x , y , z ]`?",
        "We define X=right, Y=down, Z=forward. I'm looking for the final movement as a `[ x , y , z ]` vector for the point at [ {x1} , {y1} ] in Image-1.",
        "Between these images, using X=right, Y=down, Z=forward, what's the vector that the point at [ {x1} , {y1} ] in Image-1 traveled in mm?",
        "The coordinate system is X=right, Y=down, Z=forward. Is there a 3D vector capturing the net offset from the original spot for the point at [ {x1} , {y1} ] in Image-1?",
        "In the orientation where X=right, Y=down, Z=forward, how do we write the point at [ {x1} , {y1} ] in Image-1's translation in the form `[ x_offset , y_offset , z_offset ]` mm?",
        "Assuming the first image sets X=right, Y=down, Z=forward, check the final difference in each axis-this forms a vector for the point at [ {x1} , {y1} ] in Image-1's movement, correct?",
        "Given X=right, Y=down, Z=forward, I'd like you to list the point at [ {x1} , {y1} ] in Image-1's total displacement vector `[ x , y , z ]` in mm.",
        "The first image orients X->right, Y->down, Z->forward. Please specify that vector showing exactly how the point at [ {x1} , {y1} ] in Image-1 moved in all three dimensions."
    ]
}

# add "The coordinates [ x , y ] are normalized to 0-1 and scaled by 1000, with [ 0 , 0 ] at the top-left corner. The x-axis represents the width, and the y-axis represents the height." after the question.
QUESTION_TEMPLATES["tapvid3d_total_distance"] = [question + " The coordinates [ x , y ] are normalized to 0-1 and scaled by 1000, with [ 0 , 0 ] at the top-left corner. The x-axis represents the width, and the y-axis represents the height." for question in QUESTION_TEMPLATES["tapvid3d_total_distance"]]
QUESTION_TEMPLATES["tapvid3d_displacement_vector"] = [question + " The coordinates [ x , y ] are normalized to 0-1 and scaled by 1000, with [ 0 , 0 ] at the top-left corner. The x-axis represents the width, and the y-axis represents the height." for question in QUESTION_TEMPLATES["tapvid3d_displacement_vector"]]

ANSWER_TEMPLATES = {
    "tapvid3d_total_distance": [
        "The point traveled a total of `{total_distance}` mm.",
        "In total, the point's displacement amounts to `{total_distance}` mm.",
        "Overall, the point covers about `{total_distance}` mm from start to finish.",
        "Comparing both images, the point moved roughly `{total_distance}` mm in distance.",
        "We see a net travel of `{total_distance}` mm for the point.",
        "The total distance covered by the point is `{total_distance}` mm.",
        "The point's translation spans `{total_distance}` mm.",
        "The final measure of the point's movement is `{total_distance}` mm.",
        "The complete shift in the point's position is `{total_distance}` mm.",
        "The point's overall distance of travel is `{total_distance}` mm.",
        "The point has moved a distance of `{total_distance}` mm.",
        "The point's total path length measures `{total_distance}` mm.",
        "From its initial to final position, the point traveled `{total_distance}` mm.",
        "The point's displacement magnitude is `{total_distance}` mm.",
        "The point shifted a total of `{total_distance}` mm between frames.",
        "The measured distance of the point's movement is `{total_distance}` mm.",
        "The point traversed `{total_distance}` mm in total.",
        "The point's spatial displacement equals `{total_distance}` mm.",
        "Between the two images, the point moved `{total_distance}` mm.",
        "The point's movement spans a distance of `{total_distance}` mm.",
        "The total length of the point's path is `{total_distance}` mm.",
        "The point covered a distance of `{total_distance}` mm.",
        "The point's total displacement measures `{total_distance}` mm.",
        "The point moved through a distance of `{total_distance}` mm.",
        "The point's travel distance amounts to `{total_distance}` mm.",
        "The point's total movement spans `{total_distance}` mm.",
        "The point traversed a path of `{total_distance}` mm.",
        "The point's displacement covers `{total_distance}` mm.",
        "The total movement of the point measures `{total_distance}` mm.",
        "The point's overall travel distance is `{total_distance}` mm."
    ],
    "tapvid3d_displacement_vector": [
        "The displacement vector is `[ {x_value} , {y_value} , {z_value} ]` mm.",
        "Summarily, the point's movement vector is `[ {x_value} , {y_value} , {z_value} ]` mm.",
        "Its translation can be described by `[ {x_value} , {y_value} , {z_value} ]` mm.",
        "The overall shift is `[ {x_value} , {y_value} , {z_value} ]` mm.",
        "We can represent its movement as `[ {x_value} , {y_value} , {z_value} ]` mm in 3D space.",
        "Comparing both positions, the net translation vector is `[ {x_value} , {y_value} , {z_value} ]` mm.",
        "From start to end, the point's displacement is `[ {x_value} , {y_value} , {z_value} ]` mm.",
        "The final vector describing the point's movement is `[ {x_value} , {y_value} , {z_value} ]` mm.",
        "We have `[ {x_value} , {y_value} , {z_value} ]` mm as the shift.",
        "The point's total 3D movement is `[ {x_value} , {y_value} , {z_value} ]` mm.",
        "The point moved by vector `[ {x_value} , {y_value} , {z_value} ]` mm.",
        "The point's spatial displacement vector is `[ {x_value} , {y_value} , {z_value} ]` mm.",
        "The point's position changed by `[ {x_value} , {y_value} , {z_value} ]` mm.",
        "The point shifted in space by `[ {x_value} , {y_value} , {z_value} ]` mm.",
        "The point's movement can be quantified as `[ {x_value} , {y_value} , {z_value} ]` mm.",
        "The point traversed vector `[ {x_value} , {y_value} , {z_value} ]` mm.",
        "The point's displacement coordinates are `[ {x_value} , {y_value} , {z_value} ]` mm.",
        "The point's motion vector measures `[ {x_value} , {y_value} , {z_value} ]` mm.",
        "The point traveled along vector `[ {x_value} , {y_value} , {z_value} ]` mm.",
        "The point's positional change vector is `[ {x_value} , {y_value} , {z_value} ]` mm.",
        "The point's spatial translation vector is `[ {x_value} , {y_value} , {z_value} ]` mm.",
        "The point underwent displacement `[ {x_value} , {y_value} , {z_value} ]` mm.",
        "The point's movement direction and magnitude is `[ {x_value} , {y_value} , {z_value} ]` mm.",
        "The point's spatial offset vector is `[ {x_value} , {y_value} , {z_value} ]` mm.",
        "The point's displacement in 3D space is `[ {x_value} , {y_value} , {z_value} ]` mm.",
        "The point's movement can be expressed as `[ {x_value} , {y_value} , {z_value} ]` mm.",
        "The point's spatial transformation vector is `[ {x_value} , {y_value} , {z_value} ]` mm.",
        "The point's displacement components are `[ {x_value} , {y_value} , {z_value} ]` mm.",
        "The point's movement in vector form is `[ {x_value} , {y_value} , {z_value} ]` mm.",
        "The point's spatial change vector is `[ {x_value} , {y_value} , {z_value} ]` mm."
    ]
}


class TwoFrameVideoQAEngine:
    def __init__(self, question_type, sub_dataset):
        self.question_type = question_type
        self.task_description = TASK_DESCRIPTION
        self.question_templates = QUESTION_TEMPLATES[self.question_type]
        self.answer_templates = ANSWER_TEMPLATES[self.question_type]
        self.sub_dataset = sub_dataset
        self.object_not_moving_threshold = 0.01
        self.camera_not_moving_threshold = 0.01
        self.future_frame_windows = 1e8 # for dirvetrack, due to inaccurate of extrinsics, we only consider image pairs within 5 frames.


    def project_point(self, point_3d, intrinsics, image_height, image_width, id=""):
        """
        Project 3D point into 2D normalized image coordinates using camera intrinsics.
        """
        fx, fy, cx, cy = intrinsics
        x_3d, y_3d, z_3d = point_3d
        
        # Project into 2D image coordinates
        u = (fx * x_3d / (z_3d + 1e-8)) + cx
        v = (fy * y_3d / (z_3d + 1e-8)) + cy
        
        # Normalize coordinates to [0, 1] with respect to image dimensions
        u_normalized = u / image_width
        v_normalized = v / image_height
        
        
        # assert the coordinates are valid
        # check
        if not (0 <= u_normalized < 1 and 0 <= v_normalized < 1 and z_3d > 0):
            print(f"({u_normalized}, {v_normalized}, {z_3d}) is invaid. Points3D are {point_3d}. intrinsics is {intrinsics}.")
            return None 

        return [u_normalized, v_normalized]

    def format_training_samples(self, sample_pairs, intrinsics, scene_id, points_pos_world, points_pos_cam, image_height, image_width, extrinsics_w2c):
        """
        points_pos: [#frames, #points, xyz]
        """
        sample_data = []
        entry_id = 0

        for sample_pair in sample_pairs:
            frame1, frame2 = sample_pair['frame1'], sample_pair['frame2']
            point_index = sample_pair['point_index']

            # get the positions
            position1_world = points_pos_world[frame1, point_index]
            position2_world = points_pos_world[frame2, point_index]

            displacement_vector_world = position2_world - position1_world
            displacement_distance = np.linalg.norm(displacement_vector_world)

            if displacement_distance < self.object_not_moving_threshold:
                point_moving = False
                displacement_distance = 0
                displacement_vector_world[:] = 0
            else:
                point_moving = True
            
            # if camera moving, camera2world
            E1_c2w = np.linalg.inv(extrinsics_w2c[frame1])
            E2_c2w = np.linalg.inv(extrinsics_w2c[frame2])

            camera_distance = np.linalg.norm(E2_c2w[:3, 3] - E1_c2w[:3, 3])
            if camera_distance < self.camera_not_moving_threshold:
                camera_moving = False
            else:
                camera_moving = True
            
            # need to transform the displacement from world to camera1
            # make the vector the homogeneous coordinates
            displacement_vector_world_hom = np.concatenate([displacement_vector_world, [0]]) # note, displacement vector should not consider the translation, so should use 0. Better to calculate the point's position in camera1 first.
            displacement_vector_cam1_hom = extrinsics_w2c[frame1] @ displacement_vector_world_hom
            displacement_vector_cam1 = displacement_vector_cam1_hom[:3]

            point_2d_normalized_1 = self.project_point(points_pos_cam[frame1, point_index], intrinsics, image_height, image_width, id=f"{scene_id}_f{frame1}_p{point_index}")
            point_2d_normalized_2 = self.project_point(points_pos_cam[frame2, point_index], intrinsics, image_height, image_width, id=f"{scene_id}_f{frame2}_p{point_index}")
            if point_2d_normalized_1 is None or point_2d_normalized_2 is None:
                print(f"Encounter an invalid sample. {scene_id}_f{frame1}_p{point_index} or {scene_id}_f{frame2}_p{point_index}. Skip.")
                continue

            x1, y1 = round(point_2d_normalized_1[0] * 1000), round(point_2d_normalized_1[1] * 1000)
            x2, y2 = round(point_2d_normalized_2[0] * 1000), round(point_2d_normalized_2[1] * 1000)

            task_description = random.choice(self.task_description)
            question = random.choice(self.question_templates).format(
                x1=x1, y1=y1
            )
            answer_text = random.choice(self.answer_templates).format(
                total_distance=round(displacement_distance * 1000),
                x_value=round(displacement_vector_cam1[0] * 1000),
                y_value=round(displacement_vector_cam1[1] * 1000),
                z_value=round(displacement_vector_cam1[2] * 1000)
            )
            if not point_moving:
                answer_text = "The point did not move. " + answer_text

            conversation = [
                {"from": "human", "value": f"{task_description}\n{question}"},
                {"from": "gpt", "value": answer_text},
            ]

            images = [f"{scene_id}/{frame:05d}.jpg" for frame in [frame1, frame2]]

            entry = {
                "id": f"{scene_id}_{frame1}_{frame2}_{point_index}",
                "image": images,
                "conversations": conversation,
                "height_list": [image_height] * len(images),
                "width_list": [image_width] * len(images),
                "gt_value": int(displacement_distance * 1000) if "total_distance" in self.question_type else displacement_vector_cam1.tolist(), # note vector stored in meters
                "question_type": self.question_type,
                "point_moving": int(point_moving),
                "cam_moving": int(camera_moving),
                "p1": (x1, y1),
                "p2": (x2, y2),
            }
        
            sample_data.append(entry)
            entry_id += 1

        return sample_data

    def generate_qa_training_single_scene(self, input_file, npoints_per_group=5, npairs_per_bin=1e8, img_output_dir="", augment=True, augment_ratio=1.0):
        """
        split_name should be: [pstudio | adt]
        """

        # print(f"Processing {input_file}.")
        # get meta info from input_file
        scene_id = os.path.splitext(os.path.basename(input_file))[0]

        gt_data = np.load(input_file, allow_pickle=True)
        
        # Define image output directory based on scene ID
        scene_img_dir = os.path.join(img_output_dir, scene_id)
        os.makedirs(scene_img_dir, exist_ok=True)
        
        # Load images if they do not exist in the image output directory
        image_files = [f for f in os.listdir(scene_img_dir) if f.endswith('.jpg')]

        if len(image_files) != gt_data['images_jpeg_bytes'].shape[0]:
            # Load all the images from the binary data
            print(f"Decoding and saving iamges for {scene_id}. Total images {gt_data['images_jpeg_bytes'].shape[0]}.")
            for i, frame_bytes in enumerate(gt_data['images_jpeg_bytes']):
                arr = np.frombuffer(frame_bytes, np.uint8)
                image_bgr = cv2.imdecode(arr, flags=cv2.IMREAD_UNCHANGED)
                # Save the image to the output directory
                img_file = os.path.join(scene_img_dir, f"{i:05d}.jpg")
                cv2.imwrite(img_file, image_bgr)
        else:
            # Load one image and get the shape
            image_bgr = cv2.imread(f"{scene_img_dir}/{image_files[0]}") # H, W, 3

        # get the image shape
        image_height, image_width = image_bgr.shape[:2]
        
        # other infos
        intrinsics = gt_data['fx_fy_cx_cy'] # (4, )
        tracks_xyz = gt_data['tracks_XYZ'] # (n_frames, n_points, 3)
        visibility = gt_data['visibility'] # (n_frames, n_points)
        extrinsics_w2c = gt_data.get('extrinsics_w2c', None) # (n_frames, 4, 4)
        n_frames, n_points, _ = tracks_xyz.shape
        if extrinsics_w2c is not None:
            # for all the tracks_xyz, transform to world coordinates, they are now in camera coordinates
            c2w = np.linalg.inv(extrinsics_w2c)
            points_hom = np.concatenate([tracks_xyz, np.ones((n_frames, n_points, 1))], axis=2)
            points_world_hom = np.einsum('nij,nkj->nki', c2w, points_hom)
            tracks_xyz_world = points_world_hom[..., :3]
        else:
            tracks_xyz_world = tracks_xyz.copy()
            extrinsics_w2c = np.array([np.eye(4) for _ in range(n_frames)])

        # do body segmentation
        groups = rigid_body_segmentation(tracks_xyz)
        groups = filter_large_groups(groups, min_size=5)

        # process each group
        # 1. for each group, select randomly select npoints_per_group points
        # 2. for each point, filter out those images that the points are visible
        # 3. in these images, calculate the displacement ditsance of the point between every two images
        # 4. sort and bin according to the distances
        # 5. get the number if pairs of the middle bin
        # 6. sample each bin with that number

        sample_pairs = []

        # Process each group
        for group in groups:
            # Shuffle the group and randomly select npoints_per_group points
            shuffle(group)
            selected_points = group[:npoints_per_group]

            for point_idx in selected_points:
                # Filter images where the point is visible
                visible_frames = np.where(visibility[:, point_idx])[0] # index of visible frames
                
                # If fewer than 2 visible frames, skip this point
                if len(visible_frames) < 2:
                    continue
                
                # Calculate displacements between all pairs of frames using vectorization
                frame_pairs = np.array([(i,j) for i in range(len(visible_frames)) 
                                      for j in range(i+1, len(visible_frames))])

                if len(frame_pairs) > 0:
                    frame1_indices = visible_frames[frame_pairs[:,0]]
                    frame2_indices = visible_frames[frame_pairs[:,1]]

                    # Calculate all distances at once
                    points1 = tracks_xyz_world[frame1_indices, point_idx]
                    points2 = tracks_xyz_world[frame2_indices, point_idx] 
                    dists = np.linalg.norm(points2 - points1, axis=1)
                    
                    # Create list of tuples
                    displacements = list(zip(dists, frame1_indices, frame2_indices))
                else:
                    displacements = []

                # Separate into static and moving pairs
                static_pairs = []
                moving_pairs = []
                for disp in displacements:
                    frame1, frame2 = disp[0], disp[1]
                    if frame2 > frame1 + self.future_frame_windows:
                        continue
                    if disp[0] < self.object_not_moving_threshold:  # Static threshold
                        static_pairs.append(disp)
                    else:
                        moving_pairs.append(disp)

                selected_pairs = []
                
                # For static pairs, randomly sample one pair
                if static_pairs:
                    selected_pairs.append(random.choice(static_pairs))

                # For moving pairs, use the binning strategy
                if moving_pairs:
                    # Sort moving pairs by distance
                    moving_pairs.sort(key=lambda x: x[0])
                    distances = [d[0] for d in moving_pairs]

                    # Bin the distances into 10 equal bins
                    bin_edges = np.histogram_bin_edges(distances, bins=10)
                    binned_displacements = [[] for _ in range(10)]
                    
                    # Assign each pair to a bin
                    for dist, frame1, frame2 in moving_pairs:
                        bin_idx = np.digitize(dist, bin_edges) - 1
                        bin_idx = min(bin_idx, 9)
                        binned_displacements[bin_idx].append((dist, frame1, frame2))

                    # Get the number of pairs in the middle bin
                    mid_bin_idx = 4
                    npairs_per_bin = min(len(binned_displacements[mid_bin_idx]), npairs_per_bin)
                    # at least one
                    npairs_per_bin = max(npairs_per_bin, 1)

                    # Sample pairs from each bin
                    for bin_displacements in binned_displacements:
                        if len(bin_displacements) > npairs_per_bin:
                            sampled_pairs = random.sample(bin_displacements, npairs_per_bin)
                        else:
                            sampled_pairs = bin_displacements
                        selected_pairs.extend(sampled_pairs)

                # Store the image pairs and associated metadata for training
                for dist, frame1, frame2 in selected_pairs:
                    sample_pairs.append({
                        "point_index": point_idx,
                        "frame1": frame1,
                        "frame2": frame2,
                    })
        
        if augment:
            num_samples_to_augment = int(len(sample_pairs) * augment_ratio)
            samples_to_augment = random.sample(sample_pairs, num_samples_to_augment)
            for sample in samples_to_augment:
                augmented_sample = {
                    "point_index": sample["point_index"],
                    "frame1": sample["frame2"],
                    "frame2": sample["frame1"]
                }
                sample_pairs.append(augmented_sample)

        # after getting the qa_pairs, need to format as training samples
        data = self.format_training_samples(sample_pairs, intrinsics=intrinsics, scene_id=scene_id, points_pos_world=tracks_xyz_world, points_pos_cam=tracks_xyz,
                                            image_height=image_height,
                                            image_width=image_width,
                                            extrinsics_w2c=extrinsics_w2c)
    
        return data
    
    def train_worker(self, args):
        return self.generate_qa_training_single_scene(*args)

    def generate_qa_training_data(self, scene_id_list, source_data_root, output_dir, output_file, img_output_dir, npoints_per_group, npairs_per_bin, augment, augment_ratio=1.0, max_samples=-1, num_workers=20):
        scene_files = [os.path.join(source_data_root, f"{scene_id}.npz") for scene_id in scene_id_list]

        tasks = [(scene_file, npoints_per_group, npairs_per_bin, img_output_dir, augment, augment_ratio) for scene_file in scene_files]
        with Pool(num_workers) as pool:
            all_train_data = pool.map(self.train_worker, tasks)

        train_data = [item for sublist in all_train_data for item in sublist]

        if max_samples > 0 and len(train_data) > max_samples:
            train_data = random.sample(train_data, max_samples)

        random.shuffle(train_data)

        # save to file
        with open(output_file, 'w') as f:
            for entry in train_data:
                f.write(json.dumps(entry) + '\n')

        point_not_moving_count = sum(1 for entry in train_data if entry["point_moving"] == 0)
        point_moving_count = len(train_data) - point_not_moving_count
        cam_not_moving_count = sum(1 for entry in train_data if entry["cam_moving"] == 0)
        cam_moving_count = len(train_data) - cam_not_moving_count

        # 打印信息
        print(f"Training data saved to {output_file}. In total, there are {len(train_data)} samples.")
        print(f"Object not moving: {point_not_moving_count}, Object moving: {point_moving_count}")
        print(f"Camera not moving: {cam_not_moving_count}, Camera moving: {cam_moving_count}")
    
    def format_eval_sample(self, training_sample):
        question = training_sample['conversations'][0]['value']
        training_sample['text'] = question
        return training_sample

    def generate_qa_eval_data(self, scene_id_list, source_data_root, output_dir, output_file, img_output_dir, npoints_per_group, npairs_per_bin, augment, augment_ratio=0.3, max_samples=300, num_workers=20):

        scene_files = [os.path.join(source_data_root, f"{scene_id}.npz") for scene_id in scene_id_list]

        tasks = [(scene_file, npoints_per_group, npairs_per_bin, img_output_dir, augment, augment_ratio) for scene_file in scene_files]
        with Pool(num_workers) as pool:
            all_train_data = pool.map(self.train_worker, tasks)

        train_data = [item for sublist in all_train_data for item in sublist]

        if max_samples > 0 and len(train_data) > max_samples:
            train_data = random.sample(train_data, max_samples)
        
        # need to change to eval format
        eval_data = list(map(self.format_eval_sample, train_data))

        # save to file
        with open(output_file, 'w') as f:
            for entry in eval_data:
                f.write(json.dumps(entry) + '\n')


        point_not_moving_count = sum(1 for entry in train_data if entry["point_moving"] == 0)
        point_moving_count = len(train_data) - point_not_moving_count
        cam_not_moving_count = sum(1 for entry in train_data if entry["cam_moving"] == 0)
        cam_moving_count = len(train_data) - cam_not_moving_count

        # 打印信息
        print(f"Evaluation data saved to {output_file}. In total, there are {len(eval_data)} samples.")   
        print(f"Object not moving: {point_not_moving_count}, Object moving: {point_moving_count}")
        print(f"Camera not moving: {cam_not_moving_count}, Camera moving: {cam_moving_count}")


if __name__ == "__main__":
    version = "v1_0"

    sub_datasets = ["adt", "pstudio"]
    train_question_samples = {
        "tapvid3d_total_distance": 3000000,
        "tapvid3d_displacement_vector": 3000000,
    }
    val_question_samples = {
        "tapvid3d_total_distance": 300,
        "tapvid3d_displacement_vector": 300,
    }

    train_output_dir = f"training_data/object_movement_coord/{version}"
    val_output_dir = f"evaluation_data/object_movement_coord/{version}"
    mmengine.mkdir_or_exist(train_output_dir)
    mmengine.mkdir_or_exist(val_output_dir)

    img_output_dir = "data/my_tapvid3d_images" # store all the images together for all three splits
    base_npz_file = "data/tapvid3d_dataset"
    meta_data_dir = "data/tapvid3d_dataset/meta_data"

    # generate val data
    for q_type in train_question_samples.keys():
        for sub_dataset in sub_datasets:
            print(f"Generating train data for {q_type} in {sub_dataset}.")
            
            source_data_root = f"{base_npz_file}/{sub_dataset}" # npz file dir of subdataset
            scene_id_list = mmengine.list_from_file(f"{meta_data_dir}/{sub_dataset}/val.txt") # store the train/val ids

            output_dir = f"{val_output_dir}/{sub_dataset}" # output dir
            mmengine.mkdir_or_exist(output_dir)

            train_output_file = os.path.join(output_dir, f"{sub_dataset}_{q_type}_val.jsonl")

            qa_engine = TwoFrameVideoQAEngine(question_type=q_type, sub_dataset=sub_dataset)
            qa_engine.generate_qa_eval_data(scene_id_list=scene_id_list, source_data_root=source_data_root, output_dir=output_dir, output_file=train_output_file,
                                                 img_output_dir=img_output_dir, npoints_per_group=1, npairs_per_bin=1,
                                                 augment=False, max_samples=300, num_workers=20)

    # generate train data
    for q_type in train_question_samples.keys():
        for sub_dataset in sub_datasets:
            npoints_per_group = 15
            npairs_per_bin = 30

            print(f"Generating train data for {q_type} in {sub_dataset}.")
            
            source_data_root = f"{base_npz_file}/{sub_dataset}" # npz file dir of subdataset
            scene_id_list = mmengine.list_from_file(f"{meta_data_dir}/{sub_dataset}/train.txt") # store the train/val ids

            output_dir = f"{train_output_dir}/{sub_dataset}" # output dir
            mmengine.mkdir_or_exist(output_dir)

            train_output_file = os.path.join(output_dir, f"{sub_dataset}_{q_type}_train_{npoints_per_group}points_{npairs_per_bin}pairs.jsonl")

            qa_engine = TwoFrameVideoQAEngine(question_type=q_type, sub_dataset=sub_dataset)
            qa_engine.generate_qa_training_data(scene_id_list=scene_id_list, source_data_root=source_data_root, output_dir=output_dir, output_file=train_output_file,
                                                 img_output_dir=img_output_dir, npoints_per_group=npoints_per_group, npairs_per_bin=npairs_per_bin,
                                                 augment=True, augment_ratio=0.05, num_workers=20)
