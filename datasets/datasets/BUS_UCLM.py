import logging
import argparse
import tempfile
from pathlib import Path
import shutil
import os
from typing import List, Tuple, Optional
from utils import extract_slices, CocoAnnotationObject, fill_mask_labeled
from mmengine.utils import track_iter_progress, mkdir_or_exist, scandir
from mmcv import imshow, gray2bgr, gray2rgb, rgb2gray, bgr2gray, imwrite, imread, imrotate, imresize_like, imflip, VideoReader
import numpy as np
import re
import glob
from collections import defaultdict
import json
import SimpleITK as sitk
from PIL import Image, ImageDraw

dataset_name = "BUS-UCLM"


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Convert {dataset_name} to frames & COCO style annotations")
    parser.add_argument(
        '--path',
        type=str,
        help='dataset path',
        default=f"/DATA/{dataset_name}.zip")
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
        help='whether delete  non-zipped output dataset')
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

    coco = CocoAnnotationObject(
        dataset_name=dataset_name,
        patient_id=None,
        id_procedure=None,
        save_path=Path(save_path),
        dataset_description=dataset_name,
        dataset_url="https://data.mendeley.com/datasets/7fvgj4jsp7/1"
    )
    coco.set_categories([
        {"supercategory": "nodule", "id": 1, "name": "benign_breast_lesion"},
        {"supercategory": "nodule", "id": 2, "name": "malignant_breast_lesion"},
    ])

    # Use a temporary directory to unzip the dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Unzipping dataset to temporary directory {temp_dir}")
        shutil.unpack_archive(args.path, temp_dir)
        logging.info("Done")

        img_filenames = list(scandir(temp_dir, suffix=("png",), recursive=True))

        mapping = defaultdict(lambda: defaultdict(str))
        for filename in img_filenames:
            idx = filename.split("/")[-1].split(".")[0]
            modality = "img"
            if "masks" in filename:
                modality = "mask"
            mapping[idx][modality] = f"{temp_dir}/{filename}"

        for data in mapping.values():
            print(data["img"], data["mask"])
            img = imread(data["img"])
            height, width, _ = img.shape
            rgb_mask = imread(data["mask"], channel_order="rgb")
            # [0, 255, 0], [255, 0, 0]
            # Convert to grayscale
            mask = np.zeros(rgb_mask.shape[:2], dtype=np.uint8)
            mask[(rgb_mask == [255, 0, 0]).all(axis=2)] = 2
            mask[(rgb_mask == [0, 255, 0]).all(axis=2)] = 1
            if not np.any(mask != 0):
                continue

            coco.add_annotation_from_mask_with_labels(mask)

            image_path = coco.add_image(height, width)
            imwrite(img, image_path)
            if args.save_viz:
                coco.plot_image_with_masks(coco._image_id-1)

    coco.dump_to_json()


if __name__ == '__main__':
    main()
