import logging
import argparse
import tempfile
from pathlib import Path
import shutil
import os
from typing import List, Tuple, Optional
from utils import extract_slices, CocoAnnotationObject, fill_mask_labeled
from mmengine.utils import track_iter_progress, mkdir_or_exist, scandir
from mmcv import imshow, gray2bgr, gray2rgb, rgb2gray, bgr2gray, imwrite, imread, imrotate, imresize_like, imflip, VideoReader
import numpy as np
import re
import glob
from collections import defaultdict
import json
import SimpleITK as sitk
from PIL import Image, ImageDraw

dataset_name = "brachial_plexus"


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
    view = "sagittal"

    coco = CocoAnnotationObject(
        dataset_name=dataset_name,
        patient_id=None,
        id_procedure=None,
        save_path=Path(save_path),
        dataset_description=dataset_name,
        dataset_url="https://github.com/Regional-US/brachial_plexus"
    )
    coco.set_categories([
        {"supercategory": "nerve", "id": 1, "name": "nerve_plexus"},
        {"supercategory": "needle", "id": 2, "name": "needle"},
    ])
    # name_to_id = {"jingmai": 2, "dongmai": 1, "jirouzuzhi": 3, "shenjing": 4,
    #               "jizhu": 6, "zhifang": 5}

    img_filenames = list(scandir(args.path, suffix=("mp4", "jpg"), recursive=True))

    mapping = defaultdict(lambda: defaultdict(list))
    for filename in img_filenames:
        idx = filename.split("/")[-1].split(".")[0].split("_")[0]
        modality = "vid"
        if "needle" in filename:
            modality = "mask_needle"
        if ("ac_masks" in filename) or ("P_021_09_masks" in filename):
            modality = "mask_ac"
        mapping[idx][modality].append(f"{args.path}/{filename}")

    for data in mapping.values():
        # print(data["vid"], data["mask_needle"], data["mask_ac"])
        data["mask_ac"].sort()
        data["mask_needle"].sort()
        vid = VideoReader(data["vid"][0])
        for i, img in enumerate(vid):
            print(i, data["vid"][0])
            height, width, _ = img.shape

            mask = rgb2gray(imread(data["mask_ac"][i])).astype(np.uint8)
            mask[mask < 100] = 0
            mask[mask >= 100] = 1
            coco.add_annotation_from_mask_with_labels(mask)

            if len(data["mask_needle"]) > i:
                mask_needle = rgb2gray(imread(data["mask_needle"][i])).astype(np.uint8)
                mask_needle[mask_needle < 100] = 0
                mask_needle[mask_needle >= 100] = 2
                coco.add_annotation_from_mask_with_labels(mask_needle)

            image_path = coco.add_image(height, width)
            imwrite(img, image_path)
            if args.save_viz:
                coco.plot_image_with_masks(coco._image_id-1)

    coco.dump_to_json()


if __name__ == '__main__':
    main()
