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

dataset_name = "cardiacUDC"

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
    view = 'axial'
    threshold = 0.025

    # Check if the input path ends with .zip
    assert args.path.endswith('.zip'), f"{dataset_name} should be an archive"

    # Use a temporary directory to unzip the dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Decompressing dataset to temporary directory {temp_dir}")
        shutil.unpack_archive(args.path, temp_dir)
        os.rename(f"{temp_dir}/cardiacUDC_dataset.change2zip", f"{temp_dir}/cardiacUDC_dataset.zip")
        # shutil.unpack_archive(f"{temp_dir}/cardiacUDC_dataset.zip", f"{temp_dir}")
        subprocess.run(["7z", "x", f"-o{temp_dir}", f"{temp_dir}/cardiacUDC_dataset.zip"], check=True)
        logging.info("Done")

        filenames = list(scandir(f"{temp_dir}/", suffix="nii.gz", recursive=True))
        # arrange dict by site: [(img, mask)]
        mapping = defaultdict(lambda: defaultdict(list))
        for filename in filenames:
            path_parts = filename.split("/")
            site = path_parts[1]
            patient_name = path_parts[2].split("_")[0]
            mapping[site][patient_name].append(filename)

        # per site, per patient
        for site in mapping.keys():
            for patient in mapping[site].keys():
                # NOTE mask and img are not ordered
                data = mapping[site][patient]
                # NOTE check that we have mask and img. Site R 73 desnt have masks
                if len(data) < 2:
                    continue

                coco = CocoAnnotationObject(
                    dataset_name=dataset_name,
                    patient_id=patient,
                    id_procedure=site,
                    save_path=Path(save_path),
                    dataset_description=dataset_name,
                    dataset_url="https://www.kaggle.com/datasets/xiaoweixumedicalai/cardiacudc-dataset"
                )
                coco.set_categories([
                    {"supercategory": "cardiac", "id": 1, "name": "left_ventricle"},
                    {"supercategory": "cardiac", "id": 2, "name": "left_atrium"},
                    {"supercategory": "cardiac", "id": 3, "name": "right_Atrium"},
                    {"supercategory": "cardiac", "id": 4, "name": "right_ventricle"},
                    {"supercategory": "cardiac", "id": 5, "name": "epicardium"},
                    {"supercategory": "cardiac", "id": 6, "name": "left_ventricle2"},
                ])

                if data[0].split("/")[-1].split(".")[0].split("_")[1] == "image":
                    img_path = f"{temp_dir}/{data[0]}"
                    mask_path = f"{temp_dir}/{data[1]}"
                else:
                    img_path = f"{temp_dir}/{data[1]}"
                    mask_path = f"{temp_dir}/{data[0]}"
                imgs = extract_slices(img_path, view).astype(np.uint8)
                masks = extract_slices(mask_path, view).astype(np.uint8)
                # per img extraction, some mask are empty but whatever
                for img, mask in zip(imgs, masks):
                    # skip if mask empty
                    if not np.any(mask != 0):
                        continue

                    # NOTE for "label_all_frame", only the border of the mask are marked. Need to fill them.
                    def fill_mask_labeled(mask):
                        # Identify all unique labels in the mask (excluding the background which is assumed to be zero)
                        unique_labels = np.unique(mask)
                        filled_mask = np.zeros_like(mask)

                        for label in unique_labels:
                            if label == 0:
                                continue
                            # Create a binary mask for the current label
                            binary_mask = mask == label
                            # Fill holes in the binary mask
                            filled_binary_mask = ndimage.binary_fill_holes(binary_mask)
                            # Assign the label to the filled areas in the output mask
                            filled_mask[filled_binary_mask] = label

                        return filled_mask
                    mask = fill_mask_labeled(mask)
                    # sometime, the foreground mask is very very small
                    # its a segmentation mistake (ie one pixel is labeled in the frame only)
                    # we set a treshold to skip those frames
                    proportion_non_zero = np.sum(mask != 0) / mask.size
                    if proportion_non_zero < threshold:
                        continue
                    # img are upside down
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
