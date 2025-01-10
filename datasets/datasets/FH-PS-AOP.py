import logging
import argparse
import tempfile
from pathlib import Path
import shutil
import os
from typing import List, Tuple, Optional
from scipy.ndimage import label
from utils import extract_slices, CocoAnnotationObject, keep_largest_component
from mmengine.utils import track_iter_progress, mkdir_or_exist, scandir
from mmcv import imshow, gray2bgr, gray2rgb, rgb2gray, bgr2gray, imwrite, imread, imrotate
from skimage.measure import find_contours
from skimage.morphology import convex_hull_image, erosion, dilation, square
from scipy.spatial import ConvexHull
import cv2
import numpy as np
import re
import glob
import json

dataset_name = "FH-PS-AOP"

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

    coco = CocoAnnotationObject(
        dataset_name=dataset_name,
        patient_id=None,
        id_procedure=None,
        save_path=Path(save_path),
        dataset_description=dataset_name,
        dataset_url="https://zenodo.org/records/10829116"
    )
    coco.set_categories([
        {"supercategory": "pubic_symphysis", "id": 1, "name": "pubic_symphysis"},
        {"supercategory": "fetal_head", "id": 2, "name": "fetal_head"},
    ])

    # Check if the input path ends with .zip
    assert args.path.endswith('.zip'), f"{dataset_name} should be an archive"

    # Use a temporary directory to unzip the dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Decompressing dataset to temporary directory {temp_dir}")
        shutil.unpack_archive(args.path, temp_dir)
        shutil.unpack_archive(f"{temp_dir}/Pubic Symphysis-Fetal Head Segmentation and Angle of Progression.zip", f"{temp_dir}")
        logging.info("Done")
        filenames = list(scandir(f"{temp_dir}/Pubic Symphysis-Fetal Head Segmentation and Angle of Progression/image_mha/", suffix="mha", recursive=True))

        for filename in track_iter_progress(filenames):
            img_path = f"{temp_dir}/Pubic Symphysis-Fetal Head Segmentation and Angle of Progression/image_mha/{filename}"
            mask_path = f"{temp_dir}/Pubic Symphysis-Fetal Head Segmentation and Angle of Progression/label_mha/{filename}"

            img = extract_slices(img_path, view)[0, ...].astype(np.uint8)
            mask = extract_slices(mask_path, view).astype(np.uint8)

            # masking out ot fov segmentation (heuristic)
            invalid_mask = img == 0
            
            fov = keep_largest_component(~invalid_mask)
            # erosion to seperate artefect such as colorbar etc
            fov = erosion(fov, square(12))
            fov = keep_largest_component(fov)
            fov = dilation(fov, square(12))
            # NOTE issue here
            fov_hull_mask = convex_hull_image(fov, offset_coordinates=False)
            # imshow(hull_mask.astype(np.uint8)*255)
            mask[~fov_hull_mask] = 0
            

            height, width = img.shape
            coco.add_annotation_from_mask_with_labels(mask)
            image_path = coco.add_image(height, width)
            imwrite(img, image_path)
            if args.save_viz:
                coco.plot_image_with_masks(coco._image_id-1)

        coco.dump_to_json()


if __name__ == '__main__':
    main()
