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
import pandas as pd

dataset_name = "BUSBRA"


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
        dataset_url="https://github.com/wgomezf/BUS-BRA"
    )
    coco.set_categories([
        {"supercategory": "nodule", "id": 1, "name": "breast_nodule_benign"},
        {"supercategory": "nodule", "id": 2, "name": "breast_nodule_malignant"},
    ])
    coco_test = CocoAnnotationObject(
        dataset_name=f"test.{dataset_name}",
        patient_id=None,
        id_procedure=None,
        save_path=Path(save_path),
        dataset_description=dataset_name,
        dataset_url="https://github.com/wgomezf/BUS-BRA"
    )
    coco_test.set_categories([
        {"supercategory": "nodule", "id": 1, "name": "breast_nodule_benign"},
        {"supercategory": "nodule", "id": 2, "name": "breast_nodule_malignant"},
    ])

    # Use a temporary directory to unzip the dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Decompressing dataset to temporary directory {temp_dir}")
        shutil.unpack_archive(args.path, temp_dir)
        logging.info("Done")

        path_csv = f"{temp_dir}/BUSBRA/5-fold-cv.csv"
        df = pd.read_csv(path_csv)

        filenames = list(scandir(f"{temp_dir}/BUSBRA/Images", suffix="png", recursive=True))
        # there is one to one matching in name, just replace bus with mask
        for filename in track_iter_progress(filenames):
            idx = filename.split("_")[-1]
            case_id = filename.split(".")[0]
            print(idx, filename)
            # here, based on filename, which is the ID, retrieve Patholofy and kfold
            # Find the corresponding pathology and kFold from the CSV
            row = df[df["ID"] == case_id]
            if not row.empty:
                pathology = row["Pathology"].values[0]
                label = 1
                if pathology == "malignant":
                    label = 2
                kfold = row["kFold"].values[0]
                print(f"Filename: {filename}, Pathology: {pathology}, kFold: {kfold}")
            else:
                print(f"No match found for {filename}")

            img = imread(f"{temp_dir}/BUSBRA/Images/{filename}")
            height, width, _ = img.shape
            mask = rgb2gray(imread(f"{temp_dir}/BUSBRA/Masks/mask_{idx}")).astype(np.uint8)//255*label
            
            active_coco = coco
            # test fold
            if kfold == 1:
                active_coco = coco_test

            active_coco.add_annotation_from_mask_with_labels(mask)
            image_path = active_coco.add_image(height, width)
            imwrite(img, image_path)

            if args.save_viz:
                active_coco.plot_image_with_masks(active_coco._image_id-1)

    coco.dump_to_json()
    coco_test.dump_to_json()


if __name__ == '__main__':
    main()
