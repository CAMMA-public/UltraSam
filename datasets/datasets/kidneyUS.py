import logging
import argparse
import tempfile
from pathlib import Path
import shutil
import os
from typing import List, Tuple, Optional
from utils import extract_slices, CocoAnnotationObject, fill_mask_labeled
from mmengine.utils import track_iter_progress, mkdir_or_exist, scandir
from mmcv import imshow, gray2bgr, gray2rgb, rgb2gray, bgr2gray, imwrite, imread, imrotate, imresize_like
import numpy as np
import re
import glob
from collections import defaultdict
import json
from PIL import Image, ImageDraw

dataset_name = "kidneyUS"


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
        dataset_url="https://github.com/rsingla92/kidneyUS "
    )
    coco.set_categories([
        {"supercategory": "kidney", "id": 1, "name": "kidney"},
    ])
    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Decompressing dataset to temporary directory {temp_dir}")
        shutil.unpack_archive(f"{args.path}/kidneyUS_images_14_june_2022.zip", temp_dir)
        logging.info("Done")
        # Use a temporary directory to unzip the dataset
        # /DATA/kidneyUS/kidneyUS_images_14_june_2022/kidneyUS_images_14_june_2022
        img_filenames = list(scandir(temp_dir, suffix=("png"), recursive=True))
        img_filenames = [f"{temp_dir}/{img}" for img in img_filenames]
        mask_capsule_filenames = list(scandir(f"{args.path}/labels/reviewed_masks_1/capsule", suffix=("png"), recursive=True))
        mask_capsule_filenames = [f"{args.path}/labels/reviewed_masks_1/capsule/{mask}" for mask in mask_capsule_filenames]
        mapping = defaultdict(lambda: defaultdict(str))

        for filename in img_filenames + mask_capsule_filenames:
            idx = filename.split("/")[-1].split(".")[0]
            modality = "img"
            if filename in mask_capsule_filenames:
                modality = "mask"
            mapping[idx][modality] = filename

        for data in mapping.values():
            print(data["img"], data["mask"])
            img = imread(data["img"])
            height, width, _ = img.shape
            mask = rgb2gray(imread(data["mask"])).astype(np.uint8)
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
