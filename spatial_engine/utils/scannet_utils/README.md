#### Notes
1. This doc assume the current directory is the root directory of the project (Multi-SpatialMLLM).
2. Create a directory `data/scannet` to store the downloaded data.
```bash
cd Multi-SpatialMLLM
mkdir -p data/scannet
```

#### Download ScanNet data
1. Follow the official website of [ScanNet](https://github.com/ScanNet/ScanNet) to get the download file `scannet_download.py`.
2. Specify the data types we need to download:
    ```python
    FILETYPES = [
        ".sens",
        ".aggregation.json",
        "_vh_clean.ply",
        "_vh_clean_2.0.010000.segs.json",
        "_vh_clean_2.ply",
        "_vh_clean.segs.json",
        "_vh_clean.aggregation.json",
        "_vh_clean_2.labels.ply",
        "_2d-instance.zip",
        "_2d-instance-filt.zip",
        "_2d-label.zip",
        "_2d-label-filt.zip",
    ]
    ```
3. Run the script to download the data:
    ```bash
    python spatial_engine/utils/scannet_utils/scannet_download.py -o data/scannet 
    ```
4. After downloading, you should have a folder structure like:
    ```
    Multi-SpatialMLLM
    ├── data
    │   └── scannet
    │       ├── scans
    │       │   ├── scene0000_00
    │       │   ├── ...
    │       └── ...
    ```

#### Process ScanNet data
1. Load point clouds, object point clouds, bounding boxes, etc.
    ```bash
    python spatial_engine/utils/scannet_utils/batch_load_scannet_data.py
    ```
    This will generate a file called `scenes_train_info.pkl`, which contains train and val splits. For convenience, we can split this info file to train and val (after update with image info if needed).

    ```python
    import mmengine
    ori_train_info = mmengine.load("data/scannet/scannet_instance_data/scenes_train_val_info.pkl")

    # * load train scene id, read a txt, one line with one id, mmengine does not support txt format
    train_scene_ids = mmengine.list_from_file("data/scannet/meta_data/scannetv2_train.txt")
    train_scene_ids.sort()
    val_scene_ids = mmengine.list_from_file("data/scannet/meta_data/scannetv2_val.txt")
    val_scene_ids.sort()

    # * ori_train_info is a dict, key is scene_id 
    # * according to the above train/val ids, split the info file to two separate files and save it.
    # * remember to check train + val = ori_train_info.keys()
    train_info = {k: ori_train_info[k] for k in train_scene_ids}
    val_info = {k: ori_train_info[k] for k in val_scene_ids}

    # * check
    assert len(train_info) + len(val_info) == len(ori_train_info)

    # * save
    mmengine.dump(train_info, "data/scannet/scannet_instance_data/scenes_train_info.pkl")
    mmengine.dump(val_info, "data/scannet/scannet_instance_data/scenes_val_info.pkl")
    ```

2. Load posed images.

    By default, we extract every image, if you want to export every nth image, you can set `frame_skip=n` like below.
    ```bash
    python spatial_engine/utils/scannet_utils/extract_posed_images.py --frame_skip 1
    ```
    Note that no matter what `frame_skip` is, the extracted images will still be saved with the ID like `00000.jpg`, `00001.jpg`, etc. It also exports depth images, use imageio to read them and should divide 1000 to get the real depth value in meters.

3. Update info_file with posed images information, this costs about 40 mins for if extracting all images. In this project, we only use every 5th image, so you need to set `frame_skip=5` in the script to skip images. Note that the you still need to set `frame_skip=1` in the `extract_posed_images.py` to make sure the image ids are consistent with our setting.
    ```bash
    python spatial_engine/utils/scannet_utils/update_info_file_with_images.py
    ```

    The images_info contains information like:
    ```python
        # Update the image data dictionary with this image's information
        image_data[image_id] = {
            "image_path": image_path,
            "depth_image_path": depth_image_path,
            "extrinsic_matrix": extrinsic_matrix
        }
        
        # Update the scene_info dictionary for the current scene_id
        scene_info[scene_id].update({
            "num_posed_images": num_posed_images,
            "images": image_data,
            "intrinsic_matrix": intrinsic_matrix
        })
    ```

#### Final Results
After processing, you should have a folder structure like:
```
Multi-SpatialMLLM
├── data
│   └── scannet
│       ├── meta_data
│       ├── scans
│       │   ├── scene0000_00
│       │   ├── ...
│       ├── posed_images
│       │   ├── scene0000_00
│       │   ├── ...
│       └── scannet_instance_data
│           ├── scene0000_00
│           ├── ...
│           ├── scenes_train_info.pkl
│           └── scenes_val_info.pkl
```
The `scannet_instance_data` contains the point clouds of the whole scene and each instance both with axis-aligned and not axis-aligned world coordinate.

