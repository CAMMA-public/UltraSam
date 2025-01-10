import logging
import argparse
import tempfile
from pathlib import Path
import shutil
import os
from typing import List, Tuple, Optional
from utils import extract_slices, CocoAnnotationObject, fill_mask_labeled, keep_largest_component
from mmengine.utils import track_iter_progress, mkdir_or_exist, scandir
from mmcv import imshow, gray2bgr, gray2rgb, rgb2gray, bgr2gray, imwrite, imread, imrotate, imresize_like
from skimage.morphology import convex_hull_image, erosion, dilation, square
import numpy as np
import re
import glob
from collections import defaultdict
import json
from PIL import Image, ImageDraw
import subprocess

dataset_name = "Fast-U-Net"


def unpack_rar(rar_file_path, destination_path):
    try:
        subprocess.run(['unrar', 'x', rar_file_path, destination_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while unpacking: {e}")

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
        dataset_url="https://github.com/vahidashkani/Fast-U-Net"
    )
    coco.set_categories([
        {"supercategory": "measurement", "id": 1, "name": "AC_HC"},
    ])
    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Decompressing dataset to temporary directory {temp_dir}")
        unpack_rar(f"{args.path}/Dataset/AC/AC_image1.rar", f"{temp_dir}")
        unpack_rar(f"{args.path}/Dataset/AC/AC_image2.rar", f"{temp_dir}")
        unpack_rar(f"{args.path}/Dataset/AC/AC_mask.rar", f"{temp_dir}")

        logging.info("Done")
        img_filenames = list(scandir(f"{temp_dir}", suffix=("png"), recursive=True))
        img_filenames = [f"{temp_dir}/{img_filename}" for img_filename in img_filenames]

        mapping = defaultdict(lambda: defaultdict(str))

        for filename in img_filenames:
            idx = filename.split("/")[-1].split(".")[0]
            modality = "img"
            if "mask" in filename:
                modality = "mask"
            mapping[idx][modality] = filename

        for data in mapping.values():
            print(data["img"], data["mask"])
            img = imread(data["img"])
            height, width, _ = img.shape
            mask = rgb2gray(imread(data["mask"])).astype(np.uint8)//255
            if not np.any(mask != 0):
                continue
            mask = erosion(mask, square(12))
            mask = keep_largest_component(mask)
            mask = dilation(mask, square(12))
            coco.add_annotation_from_mask_with_labels(mask)
            image_path = coco.add_image(height, width)
            imwrite(img, image_path)

            if args.save_viz:
                coco.plot_image_with_masks(coco._image_id-1)

    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Decompressing dataset to temporary directory {temp_dir}")
        unpack_rar(f"{args.path}/Dataset/HC/HC_image1.rar", f"{temp_dir}")
        unpack_rar(f"{args.path}/Dataset/HC/HC_image2.rar", f"{temp_dir}")
        unpack_rar(f"{args.path}/Dataset/HC/HC_image3.rar", f"{temp_dir}")
        unpack_rar(f"{args.path}/Dataset/HC/HC_mask.rar", f"{temp_dir}")

        logging.info("Done")
        img_filenames = list(scandir(f"{temp_dir}", suffix=("png"), recursive=True))
        img_filenames = [f"{temp_dir}/{img_filename}" for img_filename in img_filenames]

        mapping = defaultdict(lambda: defaultdict(str))

        for filename in img_filenames:
            idx = "_".join(filename.split("/")[-1].split(".")[0].split("_")[:2])
            modality = "img"
            if "mask" in filename:
                modality = "mask"
            mapping[idx][modality] = filename

        for data in mapping.values():
            print(data["img"], data["mask"])
            img = imread(data["img"])
            height, width, _ = img.shape
            mask = rgb2gray(imread(data["mask"])).astype(np.uint8)//255
            if not np.any(mask != 0):
                continue
            mask = erosion(mask, square(12))
            mask = keep_largest_component(mask)
            mask = dilation(mask, square(12))
            coco.add_annotation_from_mask_with_labels(mask)
            image_path = coco.add_image(height, width)
            imwrite(img, image_path)

            if args.save_viz:
                coco.plot_image_with_masks(coco._image_id-1)
    coco.dump_to_json()


if __name__ == '__main__':
    main()
