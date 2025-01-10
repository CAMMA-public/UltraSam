import logging
import argparse
import tempfile
from pathlib import Path
import shutil
import os
from typing import List, Tuple, Optional
from scipy import ndimage
from scipy.ndimage import label
from utils import extract_slices, CocoAnnotationObject, keep_largest_component
from mmengine.utils import track_iter_progress, mkdir_or_exist, scandir
from mmcv import imshow, gray2bgr, gray2rgb, rgb2gray, bgr2gray, imwrite, imread, imrotate, imflip
from skimage.measure import find_contours
from skimage.morphology import convex_hull_image, erosion, dilation, square
from scipy.spatial import ConvexHull
import cv2
import numpy as np
import subprocess
from collections import defaultdict
import re
import glob
import json

dataset_name = "segthy-dataset"

def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Convert {dataset_name} to frames & COCO style annotations")
    parser.add_argument(
        '--path',
        type=str,
        help='dataset path',
        default=f"/media/ameyer/Data4/{dataset_name}")
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
    view = 'axial'
    
    coco = CocoAnnotationObject(
        dataset_name=dataset_name,
        patient_id=None,
        id_procedure=None,
        save_path=Path(save_path),
        dataset_description=dataset_name,
        dataset_url="https://www.cs.cit.tum.de/camp/publications/segthy-dataset/"
    )
    coco.set_categories([
        {"supercategory": "artefact", "id": 1, "name": "thyroid"},
        {"supercategory": "artefact", "id": 2, "name": "carotid"},
        {"supercategory": "artefact", "id": 3, "name": "jugular"},
    ])
    img_filenames = list(scandir(args.path, suffix=("nii",), recursive=True))

    mapping = defaultdict(lambda: defaultdict(str))
    for filename in img_filenames:
        idx = filename.split("/")[-1].split(".")[0]
        modality = "img"
        if "label" in filename:
            modality = "mask"
        else:
            idx = idx[:-3]
        mapping[idx][modality] = f"{args.path}/{filename}"

    for data in mapping.values():
        print(data["img"], data["mask"])
        imgs = extract_slices(data["img"], view).astype(np.uint8)
        masks = extract_slices(data["mask"], view).astype(np.uint8)

        for img, mask in zip(imgs, masks):
            if not np.any(mask != 0):
                continue
            img = imflip(img, direction='diagonal')
            mask = imflip(mask, direction='diagonal')

            height, width = img.shape
            coco.add_annotation_from_mask_with_labels(mask)

            image_path = coco.add_image(height, width)
            imwrite(img, image_path)
            if args.save_viz:
                coco.plot_image_with_masks(coco._image_id-1)

    coco.dump_to_json()


if __name__ == '__main__':
    main()
