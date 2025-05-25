# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# {
#   scene_id:
#   {
# 	  num_posed_images: n
# 	  intrinsic_matrix: 4 * 4 array
# 	  images_info:
# 	  {
# 		  {image_id:05d}:
# 		  {
# 			  'image_path': 'data/scannet/posed_images/{scene_id}/{image_id:05d}'
# 			  'depth_image_path': ...png
# 			  'extrinsic_matrix': 4 * 4 array
# 		  }
# 	  }
#      object_id: * N (0-indexed)
#      {
#        "aligned_bbox": numpy.array of (7, )
#        "unaligned_bbox": numpy.array of (7, )
#        "raw_category": str
#      }
#      "axis_aligned_matrix": numpy.array of (4, 4)
#      "num_objects": int
#   }
# }

import os

import cv2
import mmengine
import numpy as np
import pandas as pd
import json
from mmengine.utils.dl_utils import TimeCounter

try:
    from .ops import project_mask_to_3d
except:
    print("ops are not imported.")

def project_points(points, K, E):
    """
    Project 3D points in homogeneous coordinates onto the image plane of a camera using its 4x4 intrinsic and extrinsic matrices.
    
    :param points: An Nx4 matrix where each row represents a 3D point in homogeneous coordinates (x, y, z, 1).
    :param K: The 4x4 intrinsic matrix of the camera, which includes the camera's focal lengths, optical center, and a row for homogeneous coordinates.
    :param E: The 4x4 extrinsic matrix of the camera that combines rotation and translation, camera to world matrix
    :return: An Nx2 matrix where each row represents the 2D image coordinates (x, y) of the projected points, obtained by normalizing the coordinates with respect to the third element.
             Additionally, a float array indicating the depth of the points.
    """
    # Invert the extrinsic matrix to transform points from world coordinates to camera coordinates
    E_inv = np.linalg.inv(E)

    # Transform points from world coordinates to camera coordinates using the extrinsic matrix
    camera_coords = E_inv @ points.T  # 4xN result from multiplying the extrinsic matrix with the Nx4 point matrix
    
    # TODO: check here, and the return arguments
    points_depth = camera_coords[2, :]
    
    # Project the camera coordinates onto the image plane using the intrinsic matrix
    image_coords = K @ camera_coords  # Results in a 4xN matrix after multiplication, this will not affect the depth
    
    # Normalize the projected points by the third row (z-coordinate in camera space) to convert to homogeneous coordinates
    image_coords /= image_coords[2, :]  # Ensures that the z-coordinate is scaled to 1
    
    # Return the image coordinates, discarding the homogeneous coordinate (z-coordinate)
    return image_coords.T[:, :2], points_depth # Transpose back to Nx4 and then slice to Nx2

class SceneInfoHandler:
    def __init__(self, info_path, posed_images_root="data/scannet/posed_images", instance_data_root="data/scannet/scannet_instance_data",
                 mask_image_root="data/scannet/scans", depth_value_scale=0.001):
        try:
            self.infos = mmengine.load(info_path)
            print(f"Data from {info_path} loaded successfully.")
        except Exception as e:
            print(f"Failed to load data from {info_path}: {e}")
            exit(1)
        self.posed_images_root = posed_images_root
        self.instance_data_root = instance_data_root
        self.mask_image_root = mask_image_root
        self.depth_value_scale = depth_value_scale
        
    def __len__(self):
        return len(self.infos)
    
    def get_sorted_keys(self):
        keys = list(self.infos.keys())
        keys.sort()

        return keys

    def get_intrinsic_matrix(self, scene_id, image_id=None):
        return self.infos[scene_id]["intrinsic_matrix"]  # N * 4mpy array

    def get_extrinsic_matrix(self, scene_id, image_id, warning=True):
        # This is the camera to world matrix
        image_id = self.convert_image_id_to_key(image_id)
        extrinsics = self.infos[scene_id]["images_info"][image_id][
            "extrinsic_matrix"
        ]  # N * 4 numpy array

        # if contains -inf or nan, then it's invalid
        if warning and np.any(np.isinf(extrinsics)) or np.any(np.isnan(extrinsics)):
            print(f"[SceneInfoHanlder] Warning: extrinsics matrix of {scene_id}: {image_id} contains inf or nan.")

        return extrinsics

    def get_extrinsic_matrix_align(self, scene_id, image_id):
        # Get the world-to-axis alignment matrix for the scene
        axis_align_matrix = self.get_world_to_axis_align_matrix(scene_id)
        
        # Get the extrinsic matrix for the specified image in the scene
        extrinsic_matrix = self.get_extrinsic_matrix(scene_id, image_id)
        
        # Assuming both matrices are numpy arrays, we can directly multiply them
        # Note: Matrix multiplication in numpy is done using the @ operator
        aligned_extrinsic_matrix = axis_align_matrix @ extrinsic_matrix
        
        return aligned_extrinsic_matrix

    def get_image_path(self, scene_id, image_id):
        image_id = self.convert_image_id_to_key(image_id)
        if image_id is None:
            return None

        return os.path.join(self.posed_images_root, scene_id, f"{image_id}.jpg")
    
    def get_image_shape(self, scene_id, image_id=None):
        # need to load one image to get the image shape from this scene to get the shape
        if image_id is None:
            image_id = self.get_all_image_ids(scene_id)[0]
        image_path = self.get_image_path(scene_id, image_id)
        image = cv2.imread(image_path)
        return image.shape[:2] # H, W

    def get_depth_image_path(self, scene_id, image_id):
        image_id = self.convert_image_id_to_key(image_id)
        if image_id is None:
            return None

        return os.path.join(self.posed_images_root, scene_id, f"{image_id}.png")
    
    def get_depth_image_shape(self, scene_id, image_id=0):
        depth_image = cv2.imread(self.get_depth_image_path(scene_id, image_id))
        
        return depth_image.shape[:2]
    
    def get_depth_image(self, scene_id, image_id):
        depth_image_path = self.get_depth_image_path(scene_id, image_id)
        depth_image = cv2.imread(depth_image_path, -1)

        return depth_image

    def convert_image_id_to_key(self, image_id):
        try:
            # try to convert image_id as int
            image_id = int(image_id)

            if image_id < 0:
                return None

            # then convert as specific format
            image_id = f"{image_id:05d}"
        except Exception as e:
            print(f"Failed to convert image_id: {image_id}: {e}")
            return None

        return image_id

    def get_world_to_axis_align_matrix(self, scene_id, image_id=None):
        return self.infos[scene_id]["axis_align_matrix"]

    def get_num_posed_images(self, scene_id):
        return self.infos[scene_id]["num_posed_images"]
    
    def get_all_image_ids(self, scene_id):
        return list(self.infos[scene_id]['images_info'].keys())
    
    def get_all_extrinsic_valid_image_ids(self, scene_id):
        all_image_ids = self.get_all_image_ids(scene_id)
        # filter out those with invalid
        all_valid_image_ids = [image_id for image_id in all_image_ids if self.is_posed_image_valid(scene_id, image_id)]

        return all_valid_image_ids
    
    def get_all_scene_ids(self):
        return list(self.infos.keys())

    def get_num_objects(self, scene_id):
        return self.infos[scene_id]["num_objects"]

    def get_object_gt_bbox(
        self, scene_id, object_id, axis_aligned=True, with_class_id=False
    ):
        if axis_aligned:
            bbox = self.infos[scene_id][object_id]["aligned_bbox"]
        else:
            bbox = self.infos[scene_id][object_id]["unaligned_bbox"]

        if not with_class_id:
            bbox = bbox[0:-1]
        return bbox

    def get_object_raw_category(self, scene_id, object_id):
        return self.infos[scene_id][object_id]["raw_category"]
    
    def get_object_height(self, scene_id, object_id):
        object_gt_bbox = self.get_object_gt_bbox(scene_id, object_id, axis_aligned=True)
        return object_gt_bbox[5]  # dz represents the height
    
    def get_object_length_axis_aligned(self, scene_id, object_id):
        """
        Return the index of axis according to with axis is used for the object length.
        The index is with respect to the object points. So it's 0, 1, 2. For ScanNet, 0 is x, 1 is y, 2 is z.
        """
        object_gt_bbox = self.get_object_gt_bbox(scene_id, object_id, axis_aligned=True)
        return 0 if object_gt_bbox[3] > object_gt_bbox[4] else 1
    
    def get_object_width_axis_aligned(self, scene_id, object_id):
        """
        Return the index of axis according to with axis is used for the object width.
        The index is with respect to the object points. So it's 0, 1, 2. For ScanNet, 0 is x, 1 is y, 2 is z.
        """
        object_gt_bbox = self.get_object_gt_bbox(scene_id, object_id, axis_aligned=True)
        return 0 if object_gt_bbox[3] < object_gt_bbox[4] else 1

    def get_object_length(self, scene_id, object_id):
        object_gt_bbox = self.get_object_gt_bbox(scene_id, object_id, axis_aligned=True)
        return max(object_gt_bbox[3], object_gt_bbox[4])  # max(dx, dy)

    def get_object_width(self, scene_id, object_id):
        object_gt_bbox = self.get_object_gt_bbox(scene_id, object_id, axis_aligned=True)
        return min(object_gt_bbox[3], object_gt_bbox[4])  # min(dx, dy)

    def get_object_volume(self, scene_id, object_id):
        object_gt_bbox = self.get_object_gt_bbox(scene_id, object_id, axis_aligned=True)
        return object_gt_bbox[3] * object_gt_bbox[4] * object_gt_bbox[5]  # dx * dy * dz

    def get_object_points_aligned(self, scene_id, object_id):
        points_path = os.path.join(self.instance_data_root, scene_id, f"object_{object_id}_aligned_points.npy")
        object_points = np.load(points_path, allow_pickle=True) 

        return object_points
    
    def get_object_point_index(self, scene_id, object_id):
        """
        This function returns the index of the object point clouds with respect to the whole scene point clouds.
        """
        instance_mask = self.get_scene_instance_mask(scene_id)
        object_index = np.where(instance_mask == object_id + 1)[0] # 1-indexed

        # to get the object point clouds: self.get_scene_points(scene_id)[object_index][:, :3]
        if len(object_index) == 0:
            print(f"[SceneInfoHanlder] Warning: {scene_id} does not have object {object_id}.")

        return object_index

    def get_scene_raw_categories(self, scene_id):
        """
        Return a list of raw categories of all objects in the scene without deduplication.
        """
        return [
            self.get_object_raw_category(scene_id, object_id)
            for object_id in range(self.get_num_objects(scene_id))
        ]

    def get_scene_points_align(self, scene_id):
        points_path = os.path.join(self.instance_data_root, scene_id, "aligned_points.npy")
        scene_points = np.load(points_path) 
        
        return scene_points

    def get_scene_points(self, scene_id):
        points_path = os.path.join(self.instance_data_root, scene_id, "unaligned_points.npy")
        scene_points = np.load(points_path) 
        
        return scene_points

    def get_point_3d_coordinates(self, scene_id, point_id, align=True):
        if align:
            scene_points = self.get_scene_points_align(scene_id)
        else:
            scene_points = self.get_scene_points(scene_id)
        return scene_points[point_id]
    
    def get_point_2d_coordinates_in_image(self, scene_id, image_id, point_id, align=True, check_visible=False, return_depth=False):
        point_3d = self.get_point_3d_coordinates(scene_id, point_id, align)
        point_3d = point_3d[:3]

        point_2d, point_depth = self.project_3d_point_to_image(scene_id, image_id, point_3d, align)

        if check_visible:
            visible_mask = self.check_point_visibility(scene_id, image_id, point_2d, point_depth)
            point_2d = point_2d[visible_mask]
            point_depth = point_depth[visible_mask]
        
        if return_depth:
            return point_2d, point_depth
        else:
            return point_2d

    def get_scene_instance_mask(self, scene_id):
        instance_mask_path = os.path.join(self.instance_data_root, scene_id, "instance_mask.npy")
        instance_mask = np.load(instance_mask_path) 

        return instance_mask
    
    def project_3d_point_to_image(self, scene_id, image_id, points_3d, align=True):
        """
        Project 3D points to 2D image plane.
        point_3d: N * 3 or (3, ) for n points or one point

        Returns:
            points_2d: N * 2, [x, y] for width and height
            points_depth: N, the depth of the points
        """
        intrinsic_matrix = self.get_intrinsic_matrix(scene_id, image_id)
        if align:
            extrinsic_matrix = self.get_extrinsic_matrix_align(scene_id, image_id) # camera to world
        else:
            extrinsic_matrix = self.get_extrinsic_matrix(scene_id, image_id) # world to camera

        # make point 3d as N * 4, sometimes point_3d is (3, ) or N * 3
        points_3d = np.expand_dims(points_3d, axis=0) if points_3d.ndim == 1 else points_3d # N, 3
        # make it N * 4
        points_3d = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])

        points_2d, points_depth = project_points(points_3d, intrinsic_matrix, extrinsic_matrix) # returns: N * 2, points_depth: N (float),

        return points_2d, points_depth
    
    def check_point_in_image_boundary(self, scene_id, points_2d):
        """
        points_2d: N * 2, [x, y] for width and height
        """
        image_height, image_width = self.get_image_shape(scene_id)
        in_bounds_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < image_width) & \
                            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image_height)
        return in_bounds_mask
    
    def check_point_visibility_by_depth(self, scene_id, image_id, points_2d, points_depth):
        """
        This function also considers points behind the camera. As for invalid depth values, they are 0, so no special treatment is needed.
        Args:
            points_2d: Nx2, [x, y] for width and height
            points_depth: N
        Returns:
            visible_mask: N, whether the point is visible in the image
        """
        depth_image = self.get_depth_image(scene_id, image_id)
        depth_height, depth_width = depth_image.shape[:2]
        image_height, image_width = self.get_image_shape(scene_id, image_id)

        scale_x = depth_width / image_width
        scale_y = depth_height / image_height

        depth_2d_x = np.round(points_2d[:, 0] * scale_x).astype(int)
        depth_2d_y = np.round(points_2d[:, 1] * scale_y).astype(int)

        depth_2d_x = np.clip(depth_2d_x, 0, depth_width - 1)
        depth_2d_y = np.clip(depth_2d_y, 0, depth_height - 1)

        depth_values = depth_image[depth_2d_y, depth_2d_x] * self.depth_value_scale

        # should larger than 0
        visible_mask = (points_depth > 0) & (points_depth < depth_values)

        return visible_mask

    def check_point_visibility(self, scene_id, image_id, points_2d, points_depth):
        """
        Args:
            points_2d: Nx2, [x, y] for width and height
            depths: N
        Returns:
            visible_mask: N, whether the point is visible in the image
        """
        in_bounds_mask = self.check_point_in_image_boundary(scene_id, points_2d)
        visible_mask = self.check_point_visibility_by_depth(scene_id, image_id, points_2d, points_depth)

        return in_bounds_mask & visible_mask
        
    def project_image_to_3d_with_mask(
        self, scene_id, image_id, mask=None, with_color=False
    ):
        intrinsic_matrix = self.get_intrinsic_matrix(scene_id, image_id)
        extrinsic_matrix = self.get_extrinsic_matrix(scene_id, image_id)
        world_to_axis_align_matrix = self.get_world_to_axis_align_matrix(scene_id)
        depth_image_path = self.get_depth_image_path(scene_id, image_id)
        if with_color:
            color_image = self.get_image_path(scene_id, image_id)
        else:
            color_image = None
        points_3d = project_mask_to_3d(
            depth_image_path,
            intrinsic_matrix,
            extrinsic_matrix,
            mask,
            world_to_axis_align_matrix,
            color_image=color_image,
        )
        return points_3d

    def is_posed_image_valid(self, scene_id, image_id):
        image_id = self.convert_image_id_to_key(image_id)
        if image_id is None:
            return False
        extrinsics = self.get_extrinsic_matrix(scene_id, image_id, warning=False)
        # if contains -inf or nan, then it's invalid
        if np.any(np.isinf(extrinsics)) or np.any(np.isnan(extrinsics)):
            return False
        else:
            return True

    def get_instance_mask(self, scene_id, image_id, target_id) -> np.ndarray:
        """
        Args:
            scene_id: str
            image_id: int
            target_id: int
        Returns:
            target_mask: np.ndarray with shape (H, W) refers to the width and height of the image.
        """
        image_id = int(image_id)
        mask_image_path = os.path.join(
            self.mask_image_root, scene_id, f"instance-filt", f"{image_id}.png"
        )

        mask_image = cv2.imread(mask_image_path, cv2.IMREAD_UNCHANGED)
        if mask_image is None:
            raise FileNotFoundError(
                f"Mask image not found at path: {mask_image_path}"
            )

        # instance_mask 0表示什么都没有，所以需要+1
        target_mask = np.where(mask_image == target_id + 1, 1, 0)

        return target_mask

class VisibilityInfoHandler:
    def __init__(self, visibility_info_path):
        self.visibility_info_path = visibility_info_path
        # read the visibility info file
        print(f"[VisibilityInfoHandler] Reading visibility info from {self.visibility_info_path}.")
        with TimeCounter(tag="read_visibility_info"):
            if self.visibility_info_path.endswith(".parquet"):
                # * the parquet file is like:
                # +---------------------------------+----------------------+
                # | key                             | values               |
                # +---------------------------------+----------------------+
                # | scene_001:image_to_points:img_1 | [0, 2, 5, 8, 10]     |
                # | scene_001:point_to_images:34    | [img_1, img_3]       |
                # | scene_002:image_to_points:img_5 | [1, 4, 7, 9]        |
                # +---------------------------------+----------------------+
                self.visibility_info = pd.read_parquet(self.visibility_info_path)
                self.info_format = "parquet"
            elif self.visibility_info_path.endswith(".pkl"):
                # {
                #   scene_id: {
                #     "image_to_points": {
                #       image_id: [point_index, ...],
                #       ...
                #     },
                #     "point_to_images": {
                #       point_index: [image_id, ...],
                #       ...
                #     }
                #   },
                #   ...
                # }
                self.visibility_info = mmengine.load(self.visibility_info_path)
                self.info_format = "pkl"
            else:
                raise ValueError(f"Unsupported file format: {self.visibility_info_path}")
        
        if self.info_format == "parquet":
            print(f"[VisibilityInfoHandler] Converting parquet file to dict.")
            self.visibility_info = self.convert_parquet_to_dict(self.visibility_info)

    @TimeCounter()
    def convert_parquet_to_dict(self, parquet_df):
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

    def get_image_to_points_info(self, scene_id, image_id):
        """
        Return:
            a list of points indices that the image can see: [point_index, ...]
        """
        if self.info_format == "parquet":
            # * keys are like f"{scene_id}:image_to_points:{image1}"
            # * values are a string
            key = f"{scene_id}:image_to_points:{image_id}"
            if key not in self.visibility_info:
                raise ValueError(f"Key {key} not found in visibility info.")
            return json.loads(self.visibility_info[key])
        elif self.info_format == "pkl":
            if scene_id not in self.visibility_info:
                raise ValueError(f"Scene {scene_id} not found in visibility info.")
            if image_id not in self.visibility_info[scene_id]["image_to_points"]:
                raise ValueError(f"Image {image_id} not found in visibility info for scene {scene_id}.")
            return self.visibility_info[scene_id]["image_to_points"][image_id]

    def get_point_to_images_info(self, scene_id, point_index):
        """
        Return:
            a list of image ids that the point can be seen in: [image_id, ...]
        """
        if self.info_format == "parquet":
            # * keys are like f"{scene_id}:point_to_images:{point_index}"
            # * values are a string
            key = f"{scene_id}:point_to_images:{point_index}"
            if key not in self.visibility_info:
                raise ValueError(f"Key {key} not found in visibility info.")
            return json.loads(self.visibility_info[key])
        elif self.info_format == "pkl":
            if scene_id not in self.visibility_info:
                raise ValueError(f"Scene {scene_id} not found in visibility info.")
            if point_index not in self.visibility_info[scene_id]["point_to_images"]:
                raise ValueError(f"Point {point_index} not found in visibility info for scene {scene_id}.")
            return self.visibility_info[scene_id]["point_to_images"][point_index]




if __name__ == "__main__":
    scene_infos = SceneInfoHandler(
        "data/scannet/scannet_instance_data/scenes_train_val_info_i_D5.pkl"
    )

