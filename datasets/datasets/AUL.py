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

dataset_name = "AUL"


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
        dataset_url="https://zenodo.org/records/7272660"
    )
    coco.set_categories([
        {"supercategory": "liver", "id": 1, "name": "liver"},
        {"supercategory": "nodule", "id": 2, "name": "mass"},
        {"supercategory": "outline", "id": 3, "name": "outline"},
    ])
    name_to_id = {item["name"]: item["id"] for item in coco.categories}

    # Use a temporary directory to unzip the dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Decompressing dataset to temporary directory {temp_dir}")
        shutil.unpack_archive(args.path, temp_dir)
        for patologie in ["Benign", "Malignant", "Normal"]:
            shutil.unpack_archive(f"{temp_dir}/{patologie}.zip", f"{temp_dir}/{patologie}")

        logging.info("Done")
        filenames = list(scandir(temp_dir, suffix=("jpg", "json"), recursive=True))

        mapping = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for filename in filenames:
            # 'Benign/Benign/image/194.jpg'
            # 'Benign/Benign/segmentation/outline/56.json'
            parts = filename.split("/")
            pathologie = parts[0]
            modality = parts[2]
            if modality == "outline":
                continue
            name = parts[-1].split(".")[0]

            mapping[pathologie][name][modality].append(filename)

        for pathologie in mapping.values():
            for name in pathologie.values():
                # read image
                img = imread(f"{temp_dir}/{name['image'][0]}")
                height, width, _ = img.shape

                for seg_path in name['segmentation']:
                    label = seg_path.split("/")[-2]
                    with open(f"{temp_dir}/{seg_path}", 'r') as file:
                        json_data = json.load(file)

                        # Convert JSON to a tuple list
                        polygon = [(point[0], point[1]) for point in json_data]

                    # Create a new image with background
                    mask = Image.new('L', (width, height), 0)

                    # Draw the polygon
                    ImageDraw.Draw(mask).polygon(polygon, outline=name_to_id[label], fill=name_to_id[label])
                    # Convert image to numpy array
                    mask = np.array(mask)
                    coco.add_annotation_from_mask_with_labels(mask)
                image_path = coco.add_image(height, width)
                imwrite(img, image_path)

                if args.save_viz:
                    coco.plot_image_with_masks(coco._image_id-1)

    coco.dump_to_json()


if __name__ == '__main__':
    main()
