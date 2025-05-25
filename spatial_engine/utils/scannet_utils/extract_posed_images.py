# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import struct
import time
import zlib
from argparse import ArgumentParser
from functools import partial

import imageio.v2 as imageio  # * to surpress warning
import mmengine
import numpy as np

COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}

COMPRESSION_TYPE_DEPTH = {
    -1: "unknown",
    0: "raw_ushort",
    1: "zlib_ushort",
    2: "occi_ushort",
}


class RGBDFrame:
    """Class for single ScanNet RGB-D image processing."""

    def load(self, file_handle):
        self.camera_to_world = np.asarray(
            struct.unpack("f" * 16, file_handle.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        self.timestamp_color = struct.unpack("Q", file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack("Q", file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.color_data = b"".join(
            struct.unpack(
                "c" * self.color_size_bytes, file_handle.read(self.color_size_bytes)
            )
        )
        self.depth_data = b"".join(
            struct.unpack(
                "c" * self.depth_size_bytes, file_handle.read(self.depth_size_bytes)
            )
        )

    def decompress_depth(self, compression_type):
        assert compression_type == "zlib_ushort"
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        assert compression_type == "jpeg"
        return imageio.imread(self.color_data)


class SensorData:
    """Class for single ScanNet scene processing.

    Single scene file contains multiple RGB-D images.
    """

    def __init__(self, filename, frame_skip):
        self.version = 4
        self.load(filename, frame_skip)

    def load(self, filename, frame_skip):
        with open(filename, "rb") as f:
            version = struct.unpack("I", f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack("Q", f.read(8))[0]
            self.sensor_name = b"".join(struct.unpack("c" * strlen, f.read(strlen)))
            self.intrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.intrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[
                struct.unpack("i", f.read(4))[0]
            ]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[
                struct.unpack("i", f.read(4))[0]
            ]
            self.color_width = struct.unpack("I", f.read(4))[0]
            self.color_height = struct.unpack("I", f.read(4))[0]
            self.depth_width = struct.unpack("I", f.read(4))[0]
            self.depth_height = struct.unpack("I", f.read(4))[0]
            self.depth_shift = struct.unpack("f", f.read(4))[0]
            num_frames = struct.unpack("Q", f.read(8))[0]
            print(f"Number of total frames: {num_frames}")
            self.frames = []

            # * use frame_skip to get index
            index = list(range(0, num_frames, frame_skip))
            for i in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)  # should iterate to get the next frame
                if i in index:
                    self.frames.append(frame)

            assert len(index) == len(self.frames), "Number of frames mismatch."
            print(f"Exported {len(index)} frames. Frame skip is {frame_skip}.")

    def export_depth_images(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for f in range(len(self.frames)):
            depth_data = self.frames[f].decompress_depth(self.depth_compression_type)
            depth = np.fromstring(depth_data, dtype=np.uint16).reshape(
                self.depth_height, self.depth_width
            )
            imageio.imwrite(
                os.path.join(output_path, self.index_to_str(f) + ".png"), depth
            )

    def export_color_images(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for f in range(len(self.frames)):
            color = self.frames[f].decompress_color(self.color_compression_type)
            imageio.imwrite(
                os.path.join(output_path, self.index_to_str(f) + ".jpg"), color
            )

    @staticmethod
    def index_to_str(index):
        return str(index).zfill(5)

    @staticmethod
    def save_mat_to_file(matrix, filename):
        with open(filename, "w") as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt="%f")

    def export_poses(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for f in range(len(self.frames)):
            self.save_mat_to_file(
                self.frames[f].camera_to_world,
                os.path.join(output_path, self.index_to_str(f) + ".txt"),
            )

    def export_intrinsics(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.save_mat_to_file(
            self.intrinsic_color, os.path.join(output_path, "intrinsic.txt")
        )


def process_scene(path, frame_skip, idx):
    """Process single ScanNet scene.

    Extract RGB images, poses and camera intrinsics.
    """
    print(f"Processing {idx}.")
    t1 = time.time()
    output_path = os.path.join("posed_images", idx)
    if mmengine.exists(output_path):
        print(f"{output_path} already exists. Skip.")
        return
    data = SensorData(os.path.join(path, idx, f"{idx}.sens"), frame_skip)
    data.export_color_images(output_path)
    data.export_intrinsics(output_path)
    data.export_poses(output_path)
    data.export_depth_images(output_path)
    print(f"Finish processing {idx}. Using {time.time() - t1}s.")


def process_directory(path, frame_skip, nproc):
    print(f"processing {path}")
    scan_ids = os.listdir(path)
    # debug
    mmengine.track_parallel_progress(
        func=partial(process_scene, path, frame_skip),
        tasks=scan_ids,
        nproc=nproc,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--frame_skip", type=int, default=1, help="export every nth frame"
    )  # * use this or --max-images-per-scene
    parser.add_argument("--nproc", type=int, default=20)
    args = parser.parse_args()

    # process train and val scenes
    if os.path.exists("scans"):
        process_directory(
            "scans", args.frame_skip, args.nproc
        )
    # process test scenes
    if os.path.exists("scans_test"):
        process_directory(
            "scans_test", args.frame_skip, args.nproc
        )
