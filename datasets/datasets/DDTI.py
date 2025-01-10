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

# NOTE archive is TN3K but DDTI is zipped inside the archive
dataset_name = "TN3K"

def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Convert DDTI to frames & COCO style annotations")
    parser.add_argument(
        '--path',
        type=str,
        help='dataset path',
        default=f"/DATA/{dataset_name}.zip")
    parser.add_argument(
        '--save-dir',
        type=str,
        help='the dir to save dataset',
        default=f"/media/ameyer/Data4/ULTRASam/datasets/DDTI")
    parser.add_argument(
        '--save-viz',
        action='store_false',
        help='save img vizualisation')
    parser.add_argument(
        '--zip',
        action='store_false',
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
        dataset_name="DDTI",
        patient_id=None,
        id_procedure=None,
        save_path=Path(save_path),
        dataset_description=dataset_name,
        dataset_url="https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/TN3K.md"
    )
    coco.set_categories([
        {"supercategory": "nodule", "id": 1, "name": "thyroid_nodule"},
    ])

    # Check if the input path ends with .zip
    assert args.path.endswith('.zip'), f"{dataset_name} should be an archive"

    # Use a temporary directory to unzip the dataset
    with tempfile.TemporaryDirectory() as temp_dir_parent:
        logging.info(f"Decompressing dataset to temporary directory {temp_dir_parent}")
        shutil.unpack_archive(args.path, temp_dir_parent)
        # NOTE archive in archive
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.unpack_archive(f"{temp_dir_parent}/Thyroid Dataset/DDTI dataset/DDTI.zip", temp_dir)
            logging.info("Done")
            # List all files in both directories
            img_dir = f"{temp_dir}/2_preprocessed_data/stage1/p_image/"
            mask_dir = f"{temp_dir}/2_preprocessed_data/stage1/p_mask/"
            img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.PNG')]
            mask_files = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.PNG')]
            img_mask_pairs = [(img, os.path.join(mask_dir, os.path.basename(img))) for img in img_files if os.path.join(mask_dir, os.path.basename(img)) in mask_files]
            for img_path, mask_path in track_iter_progress(img_mask_pairs):
                img = imread(img_path)
                mask = rgb2gray(imread(mask_path))
                mask[mask == 255] = 1
                height, width, _ = img.shape
                coco.add_annotation_from_mask_with_labels(mask)
                image_path = coco.add_image(height, width)
                imwrite(img, image_path)
                if args.save_viz:
                    coco.plot_image_with_masks(coco._image_id-1)

            coco.dump_to_json()


if __name__ == '__main__':
    main()
