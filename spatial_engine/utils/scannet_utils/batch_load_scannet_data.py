# Modified from
# https://github.com/facebookresearch/votenet/blob/master/scannet/batch_load_scannet_data.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Batch mode in loading Scannet scenes with vertices and ground truth labels
for semantic and instance segmentations.

Usage example: python ./batch_load_scannet_data.py
"""

import argparse
import os
import pickle
from multiprocessing import Pool
from os import path as osp

import numpy as np
import scannet_utils

DONOTCARE_CLASS_IDS = np.array([])
# OBJ_CLASS_IDS = np.array(
# [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
# OBJ_CLASS_IDS = np.array([])


def export(mesh_file, agg_file, seg_file, meta_file, label_map_file, test_mode=False):
    """Export original files to vert, ins_label, sem_label and bbox file.

    Args:
        mesh_file (str): Path of the mesh_file.
        agg_file (str): Path of the agg_file.
        seg_file (str): Path of the seg_file.
        meta_file (str): Path of the meta_file.
        label_map_file (str): Path of the label_map_file.
        test_mode (bool): Whether is generating test data without labels.
            Default: False.

    It returns a tuple, which contains the the following things:
        np.ndarray: Vertices of points data.
        np.ndarray: Indexes of label.
        np.ndarray: Indexes of instance.
        np.ndarray: Instance bboxes.
        dict: Map from object_id to label_id.
    """

    label_map = scannet_utils.read_label_mapping(
        label_map_file, label_from="raw_category", label_to="nyu40id"
    )
    mesh_vertices = scannet_utils.read_mesh_vertices_rgb(mesh_file)

    # Load scene axis alignment matrix
    lines = open(meta_file).readlines()
    # test set data doesn't have align_matrix
    axis_align_matrix = np.eye(4)
    for line in lines:
        if "axisAlignment" in line:
            axis_align_matrix = [
                float(x) for x in line.rstrip().strip("axisAlignment = ").split(" ")
            ]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))

    # perform global alignment of mesh vertices
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:, 0:3] = mesh_vertices[:, 0:3]
    pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
    aligned_mesh_vertices = np.concatenate([pts[:, 0:3], mesh_vertices[:, 3:]], axis=1)

    # Load semantic and instance labels
    if not test_mode:
        object_id_to_segs, label_to_segs = scannet_utils.read_aggregation(
            agg_file
        )  # * return dicts with id(int) or label(str) to lists of seg ids, object ids are 1-indexed
        seg_to_verts, num_verts = scannet_utils.read_segmentation(seg_file)
        label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)
        raw_categories = np.array([None] * num_verts)  # Array to store raw categories

        object_id_to_label_id = {}
        object_id_to_raw_category = {}
        for raw_category, segs in label_to_segs.items():
            label_id = label_map[raw_category]
            for seg in segs:
                verts = seg_to_verts[seg]
                label_ids[verts] = label_id
                raw_categories[verts] = raw_category  # Assign raw category

        instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
        for object_id, segs in object_id_to_segs.items():
            for seg in segs:
                verts = seg_to_verts[seg]
                instance_ids[verts] = object_id
                if object_id not in object_id_to_label_id:
                    object_id_to_label_id[object_id] = label_ids[verts][
                        0
                    ]  # * obj_id: int
                if object_id not in object_id_to_raw_category:
                    object_id_to_raw_category[object_id] = raw_categories[verts][
                        0
                    ]  # * obj_id: str, note, the obj_id is 1-indexed
        unaligned_bboxes, unaligned_obj_point_clouds = scannet_utils.extract_bbox(
            mesh_vertices, object_id_to_segs, object_id_to_label_id, instance_ids
        )
        aligned_bboxes, aligned_obj_point_clouds = scannet_utils.extract_bbox(
            aligned_mesh_vertices,
            object_id_to_segs,
            object_id_to_label_id,
            instance_ids,
        )
    else:
        label_ids = None
        raw_categories = None
        instance_ids = None
        unaligned_bboxes = None
        aligned_bboxes = None
        object_id_to_label_id = None
        aligned_obj_point_clouds = None
        unaligned_obj_point_clouds = None
        object_id_to_raw_category = None

    return (
        mesh_vertices,
        aligned_mesh_vertices,
        label_ids,
        raw_categories,
        instance_ids,
        unaligned_bboxes,
        aligned_bboxes,
        unaligned_obj_point_clouds,
        aligned_obj_point_clouds,
        object_id_to_raw_category,
        object_id_to_label_id,
        axis_align_matrix,
    )


def export_one_scan(
    scan_name,
    output_filename_prefix,
    max_num_point,
    label_map_file,
    scannet_dir,
    test_mode=False,
):
    if not osp.exists(output_filename_prefix):
        os.makedirs(output_filename_prefix)

    mesh_file = osp.join(scannet_dir, scan_name, scan_name + "_vh_clean_2.ply")
    agg_file = osp.join(scannet_dir, scan_name, scan_name + ".aggregation.json")
    seg_file = osp.join(
        scannet_dir, scan_name, scan_name + "_vh_clean_2.0.010000.segs.json"
    )
    # includes axisAlignment info for the train set scans.
    meta_file = osp.join(scannet_dir, scan_name, f"{scan_name}.txt")
    (
        mesh_vertices,
        aligned_mesh_vertices,
        semantic_labels,
        raw_categories,
        instance_labels,
        unaligned_bboxes,
        aligned_bboxes,
        unaligned_obj_point_clouds,
        aligned_obj_point_clouds,
        object_id_to_raw_category,
        object_id_to_label_id,
        axis_align_matrix,
    ) = export(mesh_file, agg_file, seg_file, meta_file, label_map_file, test_mode)

    if not test_mode:
        # mask = np.logical_not(np.in1d(semantic_labels, DONOTCARE_CLASS_IDS))
        # mesh_vertices = mesh_vertices[mask, :]
        # semantic_labels = semantic_labels[mask]
        # instance_labels = instance_labels[mask]
        # raw_categories = raw_categories[mask]

        num_instances = len(np.unique(instance_labels))
        print(f"Num of instances: {num_instances - 1}")

        # bbox_mask = np.in1d(unaligned_bboxes[:, -1], OBJ_CLASS_IDS) # * keep all instances
        # unaligned_bboxes = unaligned_bboxes[bbox_mask, :]
        # bbox_mask = np.in1d(aligned_bboxes[:, -1], OBJ_CLASS_IDS)
        # aligned_bboxes = aligned_bboxes[bbox_mask, :]
        assert unaligned_bboxes.shape[0] == aligned_bboxes.shape[0]
        print(f"Num of care instances: {unaligned_bboxes.shape[0]}")

    if max_num_point is not None:
        max_num_point = int(max_num_point)
        N = mesh_vertices.shape[0]
        if N > max_num_point:
            choices = np.random.choice(N, max_num_point, replace=False)
            mesh_vertices = mesh_vertices[choices, :]
            if not test_mode:
                semantic_labels = semantic_labels[choices]
                instance_labels = instance_labels[choices]
                raw_categories = raw_categories[choices]

    # Save points, semantic_labels, instance_labels as .npy files
    np.save(f"{output_filename_prefix}/unaligned_points.npy", mesh_vertices)
    np.save(f"{output_filename_prefix}/aligned_points.npy", aligned_mesh_vertices)
    scene_info = {}  # Dictionary to hold scene information

    if not test_mode:
        np.save(f"{output_filename_prefix}/semantic_mask.npy", semantic_labels)
        np.save(f"{output_filename_prefix}/instance_mask.npy", instance_labels)
        np.save(f"{output_filename_prefix}/raw_category_mask.npy", raw_categories)

        # * assert these four npy have the same length
        assert (
            len(semantic_labels)
            == len(instance_labels)
            == len(raw_categories)
            == len(mesh_vertices)
        ), "Lengths of semantic_labels, instance_labels, raw_categories, and mesh_vertices are not equal."

        # Save bounding boxes and raw category names in a dict
        for obj_id, (aligned_bbox, unaligned_bbox) in enumerate(
            zip(aligned_bboxes, unaligned_bboxes)
        ):
            raw_category_name = object_id_to_raw_category.get(
                obj_id + 1, "None"
            )  # * object_id_to_raw_category is 1 indexed
            if raw_category_name == "None":
                print(
                    f"Something wrong for the raw category name of object {obj_id} in scan {scan_name}."
                )
                exit(0)
            scene_info[obj_id] = {
                "aligned_bbox": aligned_bbox,
                "unaligned_bbox": unaligned_bbox,
                "raw_category": raw_category_name,
            }

            # * save aligned and unaligned points
            # * first check if the two types of points have the same shape

            np.save(
                f"{output_filename_prefix}/object_{obj_id}_aligned_points.npy",
                aligned_obj_point_clouds[obj_id],
            )
            np.save(
                f"{output_filename_prefix}/object_{obj_id}_unaligned_points.npy",
                unaligned_obj_point_clouds[obj_id],
            )

        scene_info["axis_align_matrix"] = axis_align_matrix
        # * store the object number
        scene_info["num_objects"] = len(aligned_bboxes)

    return {scan_name: scene_info}


def worker(args):
    (
        scan_name,
        output_filename_prefix,
        max_num_point,
        label_map_file,
        scannet_dir,
        test_mode,
    ) = args
    print("-" * 20 + f"begin for {scan_name}.")
    return export_one_scan(
        scan_name,
        output_filename_prefix,
        max_num_point,
        label_map_file,
        scannet_dir,
        test_mode,
    )


def batch_export(
    max_num_point,
    output_folder,
    scan_names_file,
    label_map_file,
    scannet_dir,
    test_mode=False,
    num_workers=20,
):
    if test_mode and not os.path.exists(scannet_dir):
        return
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    scan_names = [line.rstrip() for line in open(scan_names_file)]
    # * sort scan_names
    scan_names.sort()
    args = [
        (
            scan_name,
            osp.join(output_folder, scan_name),
            max_num_point,
            label_map_file,
            scannet_dir,
            test_mode,
        )
        for scan_name in scan_names
    ]

    all_scene_info = {}
    with Pool(num_workers) as p:
        results = p.map(worker, args)
        for result in results:
            all_scene_info.update(result)

    # Save the combined scene information
    if test_mode:
        file_name = "scenes_test_info.pkl"
    else:
        file_name = "scenes_train_val_info.pkl"
    with open(osp.join(output_folder, file_name), "wb") as f:
        pickle.dump(all_scene_info, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_num_point", default=None, help="The maximum number of the points."
    )
    parser.add_argument(
        "--output_folder",
        default="data/scannet/scannet_instance_data",
        help="output folder of the result.",
    )
    parser.add_argument(
        "--train_scannet_dir", default="scans", help="scannet data directory."
    )
    parser.add_argument(
        "--test_scannet_dir", default="scans_test", help="scannet data directory."
    )
    parser.add_argument(
        "--label_map_file",
        default="data/scannet/meta_data/scannetv2-labels.combined.tsv",
        help="The path of label map file.",
    )
    parser.add_argument(
        "--train_scan_names_file",
        default="data/scannet/meta_data/scannet_train.txt",
        help="The path of the file that stores the scan names.",
    )
    parser.add_argument(
        "--test_scan_names_file",
        default="data/scannet/meta_data/scannetv2_test.txt",
        help="The path of the file that stores the scan names.",
    )
    args = parser.parse_args()
    batch_export(
        args.max_num_point,
        args.output_folder,
        args.train_scan_names_file,
        args.label_map_file,
        args.train_scannet_dir,
        test_mode=False,
    )
    # * change output folder for test
    args.output_folder = args.output_folder.replace("scannet", "scannet_test")
    batch_export(
        args.max_num_point,
        args.output_folder,
        args.test_scan_names_file,
        args.label_map_file,
        args.test_scannet_dir,
        test_mode=True,
    )


if __name__ == "__main__":
    main()
