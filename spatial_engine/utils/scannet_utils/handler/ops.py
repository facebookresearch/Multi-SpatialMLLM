# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import cv2
import numpy as np
import open3d as o3d


def filter_images_laplacian(image_paths, threshold=100):
    res = []
    for i in range(len(image_paths)):
        if calculate_image_sharpness(image_paths[i]) >= threshold:
            res.append(image_paths[i])
    return res


def calculate_image_sharpness(image_path):
    """
    calculate the sharpness of image
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Unable to read image from: {image_path}")

    # calculate laplacion
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    sharpness_score = laplacian.var()
    return sharpness_score


def convert_to_corners(bounding_boxes: List[np.ndarray]) -> List[np.ndarray]:
    """
    Convert bounding boxes to eight corners
    Args:
        bounding_boxes: List of bounding boxes with format [cx, cy, cz, dx, dy, dz, ...]
    Returns:
        List of eight corners for each bounding box with format [[x1, y1, z1], [x2, y2, z2], ...]
    """
    corners = []
    for bbox in bounding_boxes:
        corners.append(
            np.array(
                [
                    [
                        bbox[0] - bbox[3] / 2,
                        bbox[1] - bbox[4] / 2,
                        bbox[2] - bbox[5] / 2,
                    ],
                    [
                        bbox[0] + bbox[3] / 2,
                        bbox[1] - bbox[4] / 2,
                        bbox[2] - bbox[5] / 2,
                    ],
                    [
                        bbox[0] - bbox[3] / 2,
                        bbox[1] + bbox[4] / 2,
                        bbox[2] - bbox[5] / 2,
                    ],
                    [
                        bbox[0] + bbox[3] / 2,
                        bbox[1] + bbox[4] / 2,
                        bbox[2] - bbox[5] / 2,
                    ],
                    [
                        bbox[0] - bbox[3] / 2,
                        bbox[1] - bbox[4] / 2,
                        bbox[2] + bbox[5] / 2,
                    ],
                    [
                        bbox[0] + bbox[3] / 2,
                        bbox[1] - bbox[4] / 2,
                        bbox[2] + bbox[5] / 2,
                    ],
                    [
                        bbox[0] - bbox[3] / 2,
                        bbox[1] + bbox[4] / 2,
                        bbox[2] + bbox[5] / 2,
                    ],
                    [
                        bbox[0] + bbox[3] / 2,
                        bbox[1] + bbox[4] / 2,
                        bbox[2] + bbox[5] / 2,
                    ],
                ],
                dtype=np.float32,
            )
        )
    return corners


def calculate_iou_2d(mask1, mask2):
    """
    Calculate 2D Intersection over Union (IoU) of two masks, both of which are H * W binary numpy arrays.
    """
    # Calculate the intersection of the two masks
    intersection = np.logical_and(mask1, mask2)

    # Calculate the union of the two masks
    union = np.logical_or(mask1, mask2)

    # Calculate the IoU
    # handle zero
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 0.0

    return iou


def calculate_iou_3d(box1, box2):
    """
    Calculate 3D Intersection over Union (IoU) of two 3D boxes, both of which are N * 6
    Boxes are defined by numpy arrays with [x, y, z, dx, dy, dz],
    where (x, y, z) is the center of the box, and (dx, dy, dz) are the size of the box along each axis.
    """
    # Calculate the coordinates of the intersections points
    inter_min = np.maximum(box1[:3] - box1[3:] / 2, box2[:3] - box2[3:] / 2)
    inter_max = np.minimum(box1[:3] + box1[3:] / 2, box2[:3] + box2[3:] / 2)

    # Calculate intersection volume
    inter_dim = inter_max - inter_min
    inter_volume = np.prod(inter_dim) if np.all(inter_dim > 0) else 0

    # Calculate the volume of each box
    box1_volume = np.prod(box1[3:])
    box2_volume = np.prod(box2[3:])

    # Calculate IoU
    iou = inter_volume / (box1_volume + box2_volume - inter_volume)

    return iou


def remove_statistical_outliers(point_cloud_data, nb_neighbors=20, std_ratio=1.0):
    """
    Removes statistical outliers from a point cloud data and retains all original dimensions.

    Args:
        point_cloud_data (numpy.ndarray): The input point cloud data as a NxC numpy array where C >= 3. [N, C]
        nb_neighbors (int): Number of nearest neighbors to consider for calculating average distance.
        std_ratio (float): Standard deviation ratio; points beyond this many standard deviations are considered outliers.

    Returns:
        numpy.ndarray: Filtered point cloud data with outliers removed, including all original dimensions. [n, C]
    """
    # Convert numpy array to Open3D point cloud for XYZ coordinates
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data[:, :3])

    # Perform statistical outlier removal
    clean_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)

    # Use indices to filter the original full-dimension data
    inlier_data = point_cloud_data[ind, :]

    return inlier_data


def remove_truncated_outliers(point_cloud_data, tx: float, ty: float, tz: float):
    """
    Removes statistical outliers from a point cloud data and retains all original dimensions.

    Args:
        point_cloud_data (numpy.ndarray): The input point cloud data as a NxC numpy array where C >= 3. [N, C]
        tx: Ratio of points to remove from the beginning and end of the sorted x values
        ty: Ratio of points to remove from the beginning and end of the sorted y values
        tz: Ratio of points to remove from the beginning and end of the sorted z values

    Returns:
        numpy.ndarray: Filtered point cloud data with outliers removed, including all original dimensions. [n, C]
    """

    # assert tx, ty, tz all < 0.5
    assert tx < 0.5 and ty < 0.5 and tz < 0.5, "tx, ty, tz must be less than 0.5."

    n_points = len(point_cloud_data)
    # Calculate the number of points to remove based on the given percentages
    if tx == 0 and ty == 0 and tz == 0:
        return point_cloud_data

    nx = int(tx * n_points)
    ny = int(ty * n_points)
    nz = int(tz * n_points)

    # Process x-axis
    x_sorted_indices = np.argsort(point_cloud_data[:, 0])
    valid_x_indices = x_sorted_indices[nx:-nx] if 2 * nx < n_points else np.array([])

    # Process y-axis
    y_sorted_indices = np.argsort(point_cloud_data[:, 1])
    valid_y_indices = y_sorted_indices[ny:-ny] if 2 * ny < n_points else np.array([])

    # Process z-axis
    z_sorted_indices = np.argsort(point_cloud_data[:, 2])
    valid_z_indices = z_sorted_indices[nz:-nz] if 2 * nz < n_points else np.array([])

    # Find the intersection of valid indices across all axes
    valid_indices = np.intersect1d(valid_x_indices, valid_y_indices)
    valid_indices = np.intersect1d(valid_indices, valid_z_indices)

    # Filter the original full-dimension data
    inlier_data = point_cloud_data[valid_indices]

    return inlier_data


def calculate_aabb(point_cloud_data):
    """
    Calculates the axis-aligned bounding box (AABB) of a point cloud.

    Args:
        point_cloud_data (numpy.ndarray): The input point cloud data as a NxC numpy array where C >= 3. [N, C]

    Returns:
        tuple: Contains the center of the AABB (numpy.ndarray) and the dimensions of the AABB (numpy.ndarray). [x, y, z, dx, dy, dz]
    """
    # Calculate the min and max along each column (x, y, z)
    min_corner = np.min(point_cloud_data[:, :3], axis=0)
    max_corner = np.max(point_cloud_data[:, :3], axis=0)

    # Calculate center and dimensions
    center = (max_corner + min_corner) / 2
    dimensions = max_corner - min_corner

    # Combine center and dimensions into a single array
    result = np.concatenate([center, dimensions])

    return result


def project_mask_to_3d(
    depth_image,
    intrinsic_matrix,
    extrinsic_matrix,
    mask=None,
    world_to_axis_align_matrix=None,
    color_image=None,
):
    """
    Projects a mask to 3D space using the provided depth map and camera parameters.
    Optionally appends RGB values from a color image to the 3D points. (RGB order with 0-255 range)

    Parameters:
    - depth_image (str or ndarray): Path to the depth image or a numpy array of depth values. h, w
    - intrinsic_matrix (ndarray): The camera's intrinsic matrix. 4 * 4
    - extrinsic_matrix (ndarray): The camera's extrinsic matrix. 4 * 4
    - mask (ndarray): A binary mask (zero, non-zero array) where True values indicate pixels to project, which has the same shape with color_image. H, W. Could be None, where all pixels are projected.
    - world_to_axis_align_matrix (ndarray, optional): Matrix to align the world coordinates. 4 * 4
    - color_image (str or ndarray, optional): Path to the color image or a numpy array of color values. H, W, 3

    Returns:
    - ndarray: Array of 3D coordinates, optionally with RGB values appended. All False mask will give `array([], shape=(0, C), dtype=float64)`
    """

    # Load depth image from path if it's a string
    if isinstance(depth_image, str):
        depth_image = cv2.imread(depth_image, -1)

    # Load color image from path if it's a string
    if isinstance(color_image, str):
        color_image = cv2.imread(color_image)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    if mask is None:
        mask = np.ones(color_image.shape[:2], dtype=bool)

    # Calculate scaling factors
    scale_y = depth_image.shape[0] / mask.shape[0]
    scale_x = depth_image.shape[1] / mask.shape[1]

    # Get coordinates of True values in mask
    mask_indices = np.where(mask)
    mask_y = mask_indices[0]
    mask_x = mask_indices[1]

    # Scale coordinates to match the depth image size
    # depth_y = (mask_y * scale_y).astype(int)
    # depth_x = (mask_x * scale_x).astype(int)

    # # scale and use round
    depth_y = np.round(mask_y * scale_y).astype(int)
    depth_x = np.round(mask_x * scale_x).astype(int)

    # Clip scaled coordinates to ensure they are within the image boundary
    depth_y = np.clip(depth_y, 0, depth_image.shape[0] - 1)
    depth_x = np.clip(depth_x, 0, depth_image.shape[1] - 1)

    # Extract depth values
    depth_values = (
        depth_image[depth_y, depth_x] * 0.001
    )  # Assume depth is in millimeters

    # Filter out zero depth values
    valid = depth_values > 0
    depth_values = depth_values[valid]
    mask_x = mask_x[valid]
    mask_y = mask_y[valid]

    # Construct normalized pixel coordinates
    normalized_pixels = np.vstack(
        (
            mask_x * depth_values,
            mask_y * depth_values,
            depth_values,
            np.ones_like(depth_values),
        )
    )

    # Compute points in camera coordinate system
    cam_coords = np.dot(np.linalg.inv(intrinsic_matrix), normalized_pixels)

    # Transform to world coordinates
    world_coords = np.dot(extrinsic_matrix, cam_coords)

    # Apply world-to-axis alignment if provided
    if world_to_axis_align_matrix is not None:
        world_coords = np.dot(world_to_axis_align_matrix, world_coords)

    # Append color information if color image is provided
    if color_image is not None:
        # Scale mask coordinates for the color image
        rgb_values = color_image[mask_y, mask_x]
        return np.hstack((world_coords[:3].T, rgb_values))

    return world_coords[:3].T
