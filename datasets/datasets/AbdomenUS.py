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
from PIL import Image, ImageDraw

dataset_name = "AbdomenUS"


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
        dataset_url="https://www.kaggle.com/datasets/ignaciorlando/ussimandsegm"
    )
    coco.set_categories([
        {"supercategory": "liver", "id": 1, "name": "liver"},
        {"supercategory": "kidney", "id": 2, "name": "kidney"},
        {"supercategory": "pancreas", "id": 3, "name": "pancreas"},
        {"supercategory": "vessels", "id": 4, "name": "vessels"},
        {"supercategory": "adrenals", "id": 5, "name": "adrenals"},
        {"supercategory": "gallbladder", "id": 6, "name": "gallbladder"},
        {"supercategory": "bones", "id": 7, "name": "bones"},
        {"supercategory": "spleen", "id": 8, "name": "spleen"},
    ])
    color_to_id = {
        (100, 0, 100): 1,    # liver
        (255, 255, 0): 2,    # kidney
        (0, 0, 255): 3,      # pancreas
        (255, 0, 0): 4,      # vessels
        (0, 255, 255): 5,    # adrenals
        (0, 255, 0): 6,      # gallbladder
        (255, 255, 255): 7,  # bones
        (255, 0, 255): 8     # spleen
    }

    # Use a temporary directory to unzip the dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Decompressing dataset to temporary directory {temp_dir}")
        shutil.unpack_archive(args.path, temp_dir)
        logging.info("Done")

        filenames = list(scandir(f"{temp_dir}/abdominal_US", suffix=("png", "jpg"), recursive=True))
        mapping = defaultdict(lambda: defaultdict(str))
        for filename in filenames:
            parts = filename.split("/")
            mapping[parts[-1].split(".")[0]][parts[-3]] = filename

        for data in mapping.values():
            if len(data) < 2:
                continue # no annotation provided
            # skip AUS (generated, bad quality)
            if "AUS" in f"{data['images']}":
                print(f"skipping {temp_dir}/abdominal_US/{data['images']}")
                continue
            img = imread(f"{temp_dir}/abdominal_US/{data['images']}")
            height, width, _ = img.shape
            rgb_mask = imread(f"{temp_dir}/abdominal_US/{data['annotations']}")
            mask = np.zeros(rgb_mask.shape[:2], dtype=np.uint8)
            for rgb, id_ in color_to_id.items():
                mask[(rgb_mask == rgb).all(axis=2)] = id_

            coco.add_annotation_from_mask_with_labels(mask)
            image_path = coco.add_image(height, width)
            imwrite(img, image_path)

            if args.save_viz:
                coco.plot_image_with_masks(coco._image_id-1)

    coco.dump_to_json()


if __name__ == '__main__':
    main()
