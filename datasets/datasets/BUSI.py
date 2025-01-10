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

dataset_name = "BUSI"

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
        dataset_url="https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/BUSI.md"
    )
    coco.set_categories([
        {"supercategory": "nodule", "id": 1, "name": "breast_nodule"},
    ])

    # Check if the input path ends with .zip
    assert args.path.endswith('.zip'), f"{dataset_name} should be an archive"

    # Use a temporary directory to unzip the dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Decompressing dataset to temporary directory {temp_dir}")
        shutil.unpack_archive(args.path, temp_dir)
        logging.info("Done")

        dataset_dir = f"{temp_dir}/Dataset_BUSI_with_GT"
        # Dictionary to hold image paths and their corresponding mask paths
        img_to_masks = defaultdict(list)

        # Loop through each category directory (benign, malignant, normal)
        # exclude normal as there is no masks
        for category in ["benign", "malignant"]:
            category_dir = os.path.join(dataset_dir, category)
            for filename in os.listdir(category_dir):
                if "_mask" in filename:
                    # Construct the image filename from the mask filename
                    img_filename = filename.replace("_mask", "").replace("_1", "").replace("_2", "")
                    img_path = os.path.join(category, img_filename)
                    mask_path = os.path.join(category, filename)
                    # Append mask path to the list of masks for the corresponding image
                    img_to_masks[img_path].append(mask_path)
        for img_path, mask_paths in img_to_masks.items():
            img = imread(f"{temp_dir}/Dataset_BUSI_with_GT/{img_path}")
            height, width, _ = img.shape
            for mask_path in mask_paths:
                mask = rgb2gray(imread(f"{temp_dir}/Dataset_BUSI_with_GT/{mask_path}"))
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
