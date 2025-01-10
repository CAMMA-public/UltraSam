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

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert FALLMUD to frames & COCO style annotations')
    parser.add_argument(
        '--path',
        type=str,
        help='dataset path',
        default='/DATA/FALLMUD.zip')
    parser.add_argument(
        '--save-dir',
        type=str,
        help='the dir to save dataset',
        default=f"/media/ameyer/Data4/ULTRASam/datasets/FALLMUD")
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


def natural_sort_key(s: str) -> List:
    """
    Generate a key for sorting strings containing numbers naturally,
    correctly handling numbers with leading zeros.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def find_mask_path(masks_folder: str, image_filename: str) -> Optional[str]:
    """
    Find the corresponding mask file path for an image file, ensuring the mask
    has exactly the same base name but potentially a different file extension.

    Parameters:
        masks_folder (str): The folder where mask files are stored.
        image_filename (str): The filename of the image, with or without extension.

    Returns:
        Optional[str]: The path to the corresponding mask file, if found. None otherwise.
    """
    # Extract the base name without the file extension
    base_name = os.path.splitext(image_filename)[0]
    # Prepare a pattern to match exactly the same base name with any extension
    search_pattern = os.path.join(masks_folder, f"{base_name}.*")
    # Use glob to find matching mask files
    mask_files = glob.glob(search_pattern)

    # Filter the results to exclude any files where the base name part before the extension does not exactly match
    exact_matches = [f for f in mask_files if os.path.splitext(os.path.basename(f))[0] == base_name]

    # Return the first exactly matching mask file path if any matches are found
    return exact_matches[0] if exact_matches else None

def get_image_mask_triplets(base_folder: str) -> List[Tuple[str, str, str]]:
    images_folder = os.path.join(base_folder, "images")
    aponeurosis_masks_folder = os.path.join(base_folder, "aponeurosis_masks")
    fascicle_masks_folder = os.path.join(base_folder, "fascicle_masks")

    triplets = []

    # Sort the list of filenames naturally
    for image_filename in sorted(os.listdir(images_folder), key=natural_sort_key):
        img_path = os.path.join(images_folder, image_filename)

        aponeurosis_mask_path = find_mask_path(aponeurosis_masks_folder, image_filename)
        fascicle_mask_path = find_mask_path(fascicle_masks_folder, image_filename)

        if aponeurosis_mask_path and fascicle_mask_path:
            triplets.append((img_path, aponeurosis_mask_path, fascicle_mask_path))

    return triplets

def main() -> None:
    args = parse_args()
    save_path = Path(args.save_dir)
    mkdir_or_exist(save_path)
    mkdir_or_exist(save_path / "images")
    mkdir_or_exist(save_path / "annotations")
    if args.save_viz:
        mkdir_or_exist(save_path / "vizualisation")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    dataset_name = "FALLMUD"
    coco = CocoAnnotationObject(
        dataset_name=dataset_name,
        patient_id=None,
        id_procedure=None,
        save_path=Path(save_path),
        dataset_description=dataset_name,
        dataset_url="https://kalisteo.cea.fr/index.php/fallmud/#"
    )
    coco.set_categories([
        {"supercategory": "muscle", "id": 1, "name": "aponeurosis"},
        {"supercategory": "muscle", "id": 2, "name": "fascicle"},
    ])

    # Check if the input path ends with .zip
    assert args.path.endswith('.zip'), "FALLMUD should be an archive"

    # Use a temporary directory to unzip the dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Unzipping dataset to temporary directory {temp_dir}")
        shutil.unpack_archive(args.path, temp_dir)
        logging.info("Done")
        
        centers = ["NeilCronin", "RyanCunningham"]
        triplets = []
        for center in centers:
            base_folder = f"{temp_dir}/FALLMUD/{center}"
            triplets += get_image_mask_triplets(base_folder)

        # for each triplets (img, maskA, maskB) extract segmentations mask
        for img_path, aponeurosis_mask_path, fascicle_mask_path in track_iter_progress(triplets):
            img = imread(img_path)
            aponeurosis_mask = rgb2gray(imread(aponeurosis_mask_path))
            fascicle_mask = rgb2gray(imread(fascicle_mask_path))
            assert img.shape[-1] == 3, "should have 3 channels"

            # NOTE aponeurosis mask from RyanCunningham are rotated for some reason
            if aponeurosis_mask.shape[:2] != img.shape[:2]:
                aponeurosis_mask = imrotate(aponeurosis_mask, 90, auto_bound=True)       

            # label is somethime compressed with poor interpolation,
            # resulting in range of value 0 -> 255 so we binarize
            aponeurosis_mask[aponeurosis_mask < 100] = 0
            aponeurosis_mask[aponeurosis_mask >= 100] = 1

            fascicle_mask[fascicle_mask < 100] = 0
            # fascicle_mask[fascicle_mask > 100] = 2
            fascicle_mask[fascicle_mask >= 100] = 0

            # we have img and mask, we can create coco
            height, width, _ = img.shape
            for mask in [aponeurosis_mask, fascicle_mask]:                
                coco.add_annotation_from_mask_with_labels(mask)
            image_path = coco.add_image(height, width)
            imwrite(img, image_path)

            coco.plot_image_with_masks(coco._image_id-1)

        coco.dump_to_json()




if __name__ == '__main__':
    main()
