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
import json

dataset_name = "GIST514-DB"

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
        dataset_url="https://github.com/howardchina/query2"
    )
    coco.set_categories([
        {"supercategory": "lmym", "id": 1, "name": "lmym"},
        {"supercategory": "gist", "id": 2, "name": "gist"},
    ])
    
    test_coco = CocoAnnotationObject(
        dataset_name=f"test.{dataset_name}",
        patient_id=None,
        id_procedure=None,
        save_path=Path(save_path),
        dataset_description=dataset_name,
        dataset_url="https://github.com/howardchina/query2"
    )
    test_coco.set_categories([
        {"supercategory": "lmym", "id": 1, "name": "lmym"},
        {"supercategory": "gist", "id": 2, "name": "gist"},
    ])

    # Check if the input path ends with .zip
    assert args.path.endswith('.zip'), f"{dataset_name} should be an archive"

    # Use a temporary directory to unzip the dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Decompressing dataset to temporary directory {temp_dir}")
        shutil.unpack_archive(args.path, temp_dir)
        logging.info("Done")

        json_path = f"{temp_dir}/usd514_jpeg_roi/annotations/train_anno_crop_split_0.json"
        img_dir = f"{temp_dir}/usd514_jpeg_roi/images/"
        img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
        filenames_to_match = {path.split('/')[-1]: path for path in img_files}

        with open(json_path, 'r') as f:
            coco_data = json.load(f)

        for image in track_iter_progress(coco_data['images']):
            filename = image['file_name']
            if filename in filenames_to_match:
                corresponding_img_path = filenames_to_match[filename]
                img = imread(corresponding_img_path)
                height, width, _ = img.shape

                # For each matched image, find corresponding annotations by image id
                image_annotations = [anno for anno in coco_data['annotations'] if anno['image_id'] == image['id']]
                for annotation in image_annotations:
                    coco.add_annotation_from_polyline(annotation['segmentation'], annotation['category_id'], height, width)

                image_path = coco.add_image(height, width)
                imwrite(img, image_path)

                if args.save_viz:
                    coco.plot_image_with_masks(coco._image_id-1)

        coco.dump_to_json()


        json_path = f"{temp_dir}/usd514_jpeg_roi/annotations/val_anno_crop_split_0.json"
        img_dir = f"{temp_dir}/usd514_jpeg_roi/images/"
        img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
        filenames_to_match = {path.split('/')[-1]: path for path in img_files}

        with open(json_path, 'r') as f:
            coco_data = json.load(f)

        for image in track_iter_progress(coco_data['images']):
            filename = image['file_name']
            if filename in filenames_to_match:
                corresponding_img_path = filenames_to_match[filename]
                img = imread(corresponding_img_path)
                height, width, _ = img.shape

                # For each matched image, find corresponding annotations by image id
                image_annotations = [anno for anno in coco_data['annotations'] if anno['image_id'] == image['id']]
                for annotation in image_annotations:                
                    test_coco.add_annotation_from_polyline(annotation['segmentation'], annotation['category_id'], height, width)

                image_path = test_coco.add_image(height, width)
                imwrite(img, image_path)

                if args.save_viz:
                    test_coco.plot_image_with_masks(test_coco._image_id-1)

        test_coco.dump_to_json()


if __name__ == '__main__':
    main()
