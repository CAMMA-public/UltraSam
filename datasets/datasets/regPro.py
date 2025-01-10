import logging
import argparse
import tempfile
from pathlib import Path
import shutil
import os
from typing import List, Tuple, Optional
from utils import extract_slices, CocoAnnotationObject, fill_mask_labeled
from mmengine.utils import track_iter_progress, mkdir_or_exist, scandir
from mmcv import imshow, gray2bgr, gray2rgb, rgb2gray, bgr2gray, imwrite, imread, imrotate, imresize_like, imflip
import numpy as np
import re
import glob
from collections import defaultdict
import json
import SimpleITK as sitk
from PIL import Image, ImageDraw

dataset_name = "regPro"


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
    view = "sagittal"

    coco = CocoAnnotationObject(
        dataset_name=dataset_name,
        patient_id=None,
        id_procedure=None,
        save_path=Path(save_path),
        dataset_description=dataset_name,
        dataset_url="https://muregpro.github.io/data.html "
    )
    coco.set_categories([
        {"supercategory": "unknow1", "id": 1, "name": "unknow1"},
        # {"supercategory": "unknow2", "id": 2, "name": "unknow2"},
        # {"supercategory": "unknow3", "id": 3, "name": "unknow3"},
        # {"supercategory": "unknow4", "id": 4, "name": "unknow4"},
        # {"supercategory": "unknow5", "id": 5, "name": "unknow5"},
        # {"supercategory": "unknow6", "id": 6, "name": "unknow6"},
    ])
    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Decompressing dataset to temporary directory {temp_dir}")
        shutil.unpack_archive(args.path, temp_dir)
        shutil.unpack_archive(f"{temp_dir}/train.zip", temp_dir)
        shutil.unpack_archive(f"{temp_dir}/val.zip", temp_dir)
        logging.info("Done")
        
        # exlcude "mr_images/labels"
        img_filenames = list(scandir(f"{temp_dir}", suffix=("nii.gz"), recursive=True))
        img_filenames = [f"{temp_dir}/{img_filename}" for img_filename in img_filenames if "mr" not in img_filename]

        mapping = defaultdict(lambda: defaultdict(str))
        for filename in img_filenames:
            idx = filename.split("/")[-1].split(".")[0]
            modality = "img"
            if "labels" in filename:
                modality = "mask"
            mapping[idx][modality] = filename

        for data in mapping.values():
            print(data["img"], data["mask"])
            imgs = extract_slices(data["img"], view)
            sitk_img = sitk.ReadImage(data["mask"])
            masks = sitk.GetArrayFromImage(sitk_img)

            masks = np.transpose(masks, (3, 0, 1, 2))

            for img, mask_img in zip(imgs, masks):
                if not np.any(mask_img != 0):
                    continue
                img = imflip(img, direction='diagonal')
                height, width = img.shape

                for mask_per_label in mask_img:
                    if not np.any(mask_per_label != 0):
                        continue
                    mask_per_label[mask_per_label != 0] = 1
                    mask_per_label = imflip(mask_per_label, direction='diagonal')
                    coco.add_annotation_from_mask_with_labels(mask_per_label)
                image_path = coco.add_image(height, width)
                imwrite(img, image_path)

                if args.save_viz:
                    coco.plot_image_with_masks(coco._image_id-1)

        coco.dump_to_json()


if __name__ == '__main__':
    main()
