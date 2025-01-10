import logging
import argparse
import tempfile
from pathlib import Path
import shutil
import os
from typing import List, Tuple, Optional
from utils import extract_slices, CocoAnnotationObject
from mmengine.utils import track_iter_progress, mkdir_or_exist, scandir
from mmcv import imshow, gray2bgr, gray2rgb, rgb2gray, bgr2gray, imwrite, imread, imrotate
import numpy as np
import re
import glob
from collections import defaultdict
import json

dataset_name = "ultrasound-nerve-segmentation"

def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Convert {dataset_name} to frames & COCO style annotations")
    parser.add_argument(
        '--path',
        type=str,
        help='dataset path',
        default=f"/DATA/{dataset_name}")
    parser.add_argument(
        '--save-dir',
        type=str,
        help='the dir to save dataset',
        default=f"/media/ameyer/Data4/ULTRASam/datasets/{dataset_name}")
    parser.add_argument(
        '--save-viz',
        action='store_false',
        help='save img vizualisation')
    parser.add_argument(
        '--zip',
        action='store_true',
        help='whether zip output dataset')
    parser.add_argument(
        '--delete',
        action='store_true',
        help='whether delete non-zipped output dataset')
    args = parser.parse_args()
    return args

def main() -> None:
    args = parse_args()
    save_path = Path(args.save_dir)
    mkdir_or_exist(save_path)
    mkdir_or_exist(save_path / "images")
    mkdir_or_exist(save_path / "annotations")
    if args.save_viz:
        mkdir_or_exist(save_path / "vizualisation")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 1_1.tif    <patient_id>_<frame_id>
    # 1_1_mask.tif    <patient_id>_<frame_id>_mask
    # not an archive
    file_paths = list(scandir(f"{args.path}/train", suffix="tif", recursive=True))
    video_frame_mask_paths = defaultdict(list)

    # if not mask, add pair of path to video_id key
    for file_path in file_paths:
        is_mask = "mask" in file_path
        if not is_mask:
            continue
        path_parts = file_path.split(".")[0].split("_")
        patient_id = path_parts[0]
        frame_id = path_parts[1]
        video_frame_mask_paths[patient_id].append((f"{patient_id}_{frame_id}.tif", f"{patient_id}_{frame_id}_mask.tif"))

    for video_id, video_frame_mask_path in video_frame_mask_paths.items():
        sorted_video_frame_mask_path = sorted(video_frame_mask_path, key=lambda x: int(x[0].split('_')[1].split('.')[0]))
        coco = CocoAnnotationObject(
            dataset_name=dataset_name,
            patient_id=video_id,
            id_procedure=None,
            save_path=Path(save_path),
            dataset_description=dataset_name,
            dataset_url="https://www.kaggle.com/c/ultrasound-nerve-segmentation/data"
        )
        coco.set_categories([
            {"supercategory": "cervical_nerves", "id": 1, "name": "cervical_nerves"},
        ])
        # load img and mask per frame
        for img_path, mask_path in track_iter_progress(sorted_video_frame_mask_path):
            img = imread(f"{args.path}/train/{img_path}")
            mask = rgb2gray(imread(f"{args.path}/train/{mask_path}"))

            mask[mask < 150] = 0
            mask[mask >= 150] = 1

            height, width, _ = img.shape
            coco.add_annotation_from_mask_with_labels(mask)
            image_path = coco.add_image(height, width)
            imwrite(img, image_path)
            if args.save_viz:
                coco.plot_image_with_masks(coco._image_id-1)
        coco.dump_to_json()


if __name__ == '__main__':
    main()
