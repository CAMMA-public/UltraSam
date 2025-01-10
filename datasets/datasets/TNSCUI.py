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

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert TNSCUI to frames & COCO style annotations')
    parser.add_argument(
        '--path',
        type=str,
        help='dataset path',
        default='/DATA/tnscui.tar.gz')
    parser.add_argument(
        '--save-dir',
        type=str,
        help='the dir to save dataset',
        default='/media/ameyer/Data4/ULTRASam/datasets//TNSCUI')
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

    dataset_name = "TNSCUI"
    coco = CocoAnnotationObject(
        dataset_name=dataset_name,
        patient_id=None,
        id_procedure=None,
        save_path=Path(save_path),
        dataset_description=dataset_name,
        dataset_url="https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/TN-SCUI2020.md"
    )
    coco.set_categories([
        {"supercategory": "nodule", "id": 1, "name": "thyroid_nodule"},
    ])

    # Check if the input path ends with .zip
    assert args.path.endswith('.gz'), f"{dataset_name} should be an archive"

    # Use a temporary directory to unzip the dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Decompressing dataset to temporary directory {temp_dir}")
        shutil.unpack_archive(args.path, temp_dir)
        logging.info("Done")

        # List all files in both directories
        img_dir = f"{temp_dir}/tnscui/JPEGImages/"
        mask_dir = f"{temp_dir}/tnscui/SegmentationClass/"
        img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
        mask_files = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')]

        # Match img files with their corresponding mask files
        img_mask_pairs = [(img, os.path.join(mask_dir, os.path.splitext(os.path.basename(img))[0] + '.png')) for img in img_files if os.path.join(mask_dir, os.path.splitext(os.path.basename(img))[0] + '.png') in mask_files]

        for img_path, mask_path in track_iter_progress(img_mask_pairs):
            img = imread(img_path)
            mask = rgb2gray(imread(mask_path))
            height, width, _ = img.shape
            coco.add_annotation_from_mask_with_labels(mask)
            image_path = coco.add_image(height, width)
            imwrite(img, image_path)
            if args.save_viz:
                coco.plot_image_with_masks(coco._image_id-1)

        coco.dump_to_json()

if __name__ == '__main__':
    main()
