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

dataset_name = "MMOTU_2d"


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
        dataset_url="https://github.com/cv516Buaa/MMOTU_DS2Net"
    )
    # note is the cls file it start at 0
    coco.set_categories([
        {"supercategory": "nodule", "id": 1, "name": "chocolate_cyst"},
        {"supercategory": "nodule", "id": 2, "name": "serous_cystadenoma"},
        {"supercategory": "nodule", "id": 3, "name": "teratoma"},
        {"supercategory": "nodule", "id": 4, "name": "thera_cell_tumor"},
        {"supercategory": "nodule", "id": 5, "name": "simple_cyst"},
        {"supercategory": "nodule", "id": 6, "name": "normal_ovary"},
        {"supercategory": "nodule", "id": 7, "name": "mucinous_cystadenoma"},
        {"supercategory": "nodule", "id": 8, "name": "high_grade_serous"},
    ])

    test_coco = CocoAnnotationObject(
        dataset_name=f"test.{dataset_name}",
        patient_id=None,
        id_procedure=None,
        save_path=Path(save_path),
        dataset_description=dataset_name,
        dataset_url="https://github.com/cv516Buaa/MMOTU_DS2Net"
    )
    # note is the cls file it start at 0
    test_coco.set_categories([
        {"supercategory": "nodule", "id": 1, "name": "chocolate_cyst"},
        {"supercategory": "nodule", "id": 2, "name": "serous_cystadenoma"},
        {"supercategory": "nodule", "id": 3, "name": "teratoma"},
        {"supercategory": "nodule", "id": 4, "name": "thera_cell_tumor"},
        {"supercategory": "nodule", "id": 5, "name": "simple_cyst"},
        {"supercategory": "nodule", "id": 6, "name": "normal_ovary"},
        {"supercategory": "nodule", "id": 7, "name": "mucinous_cystadenoma"},
        {"supercategory": "nodule", "id": 8, "name": "high_grade_serous"},
    ])
    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Decompressing dataset to temporary directory {temp_dir}")
        shutil.unpack_archive(args.path, temp_dir)
        logging.info("Done")

        train_file = f"{temp_dir}/OTU_2d/train_cls.txt"
        val_file = f"{temp_dir}/OTU_2d/val_cls.txt"
        set_mapping = defaultdict(str)
    
        # Load train set
        with open(train_file, 'r') as train:
            for line in train:
                image, label = line.strip().split()
                set_mapping[image] = {'set': 'train', 'label': label}
        
        # Load val set
        with open(val_file, 'r') as val:
            for line in val:
                image, label = line.strip().split()
                set_mapping[image] = {'set': 'val', 'label': label}


        # Use a temporary directory to unzip the dataset
        filenames = list(scandir(f"{temp_dir}/OTU_2d", suffix=("PNG", "JPG"), recursive=True))
        mapping = defaultdict(lambda: defaultdict(str))

        for filename in filenames:
            if "_" in filename.split("/")[-1]:
                continue
            idx = filename.split("/")[-1].split(".")[0]
            modality = "img"
            if "annotations" in filename:
                modality = "mask"
            mapping[idx][modality] = f"{temp_dir}/OTU_2d/{filename}"

        for data in mapping.values():
            print(data["img"], data["mask"])
            img_filename = data["img"].split("/")[-1]  # Extract the image filename
            img_key = img_filename.split(".")[0] + ".JPG"  # Adjust extension if neede
            # Check if the image is in the train or val set and get its label
            if img_key in set_mapping:
                set_type = set_mapping[img_key]['set']
                label = int(set_mapping[img_key]['label']) + 1  # convert to 1 based index from 0 based index
                print(f"{data['img']} is in the {set_type} set with label {label}")
            else:
                print(f"{data['img']} not found in train or val set")
            
            # Here find out if we are in train or val
            # find out the label
            img = imread(data["img"])
            height, width, _ = img.shape
            mask = rgb2gray(imread(data["mask"])).astype(np.uint8)
            mask[mask != 0] = 1 * label

            current_coco = coco
            if set_type != "train":
                current_coco = test_coco

            current_coco.add_annotation_from_mask_with_labels(mask)
            image_path = current_coco.add_image(height, width)
            imwrite(img, image_path)

            if args.save_viz:
                current_coco.plot_image_with_masks(current_coco._image_id-1)

    coco.dump_to_json()
    test_coco.dump_to_json()


if __name__ == '__main__':
    main()
