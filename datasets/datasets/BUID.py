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

dataset_name = "BUID"

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
        dataset_url="https://qamebi.com/breast-ultrasound-images-database/"
    )
    coco.set_categories([
        {"supercategory": "nodule", "id": 1, "name": "breast_nodule"},
    ])

    # Use a temporary directory to unzip the dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Decompressing dataset to temporary directory {temp_dir}")
        shutil.unpack_archive(f"{args.path}/Benign.zip", temp_dir)
        shutil.unpack_archive(f"{args.path}/Malignant.zip", temp_dir)
        logging.info("Done")

        dataset_dir = f"{temp_dir}/"
        # List to hold pairs of image BMP paths and their corresponding mask TIF paths
        img_mask_pairs = []

        # Loop through each category directory (Benign, Malignant)
        for category in ["Benign", "Malignant"]:
            category_dir = os.path.join(dataset_dir, category)
            for filename in os.listdir(category_dir):
                if "Image.bmp" in filename:
                    img_path = os.path.join(category, filename)
                    mask_filename = filename.replace("Image.bmp", "Mask.tif")
                    mask_path = os.path.join(category, mask_filename)
                    img_mask_pairs.append((img_path, mask_path))

        for img_path, mask_path in track_iter_progress(img_mask_pairs):
            img = imread(f"{temp_dir}/{img_path}")
            height, width, _ = img.shape

            mask = rgb2gray(imread(f"{temp_dir}/{mask_path}"))
            mask[mask < 150] = 0
            mask[mask >= 150] = 1
            coco.add_annotation_from_mask_with_labels(mask)
            image_path = coco.add_image(height, width)
            imwrite(img, image_path)

            if args.save_viz:
                coco.plot_image_with_masks(coco._image_id-1)

        coco.dump_to_json()


if __name__ == '__main__':
    main()
