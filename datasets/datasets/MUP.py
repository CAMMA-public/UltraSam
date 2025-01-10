import logging
import argparse
import tempfile
from pathlib import Path
import shutil
import re
from utils import extract_slices, CocoAnnotationObject
from mmengine.utils import track_iter_progress, mkdir_or_exist, scandir
from mmcv import imshow, gray2bgr, gray2rgb, imwrite, imflip
import numpy as np
import SimpleITK as sitk

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MUP to frames & COCO style annotations')
    parser.add_argument(
        '--path',
        type=str,
        help='dataset path',
        default='/DATA/MUP.zip')
    parser.add_argument(
        '--save-dir',
        type=str,
        help='the dir to save dataset',
        default='/media/ameyer/Data4/ULTRASam/datasets/MUP')
    parser.add_argument(
        '--save-vid',
        action='store_false',
        help='save video vizualisation')
    parser.add_argument(
        '--unzip',
        action='store_true',
        help='whether unzip dataset or not, needed for zipped dataset')
    parser.add_argument(
        '--zipout',
        action='store_true',
        help='whether zip the resulting dataset and delete unzipped artifacts')
    args = parser.parse_args()
    return args



def create_background_mask(image: np.ndarray) -> np.ndarray:
    """
    Create a mask for a 3D image where any voxel is considered background if any
    one of its dimensions is fully composed of zeros.

    Parameters:
    - image: A 3D numpy array representing the image.

    Returns:
    - A 3D numpy array (mask) of the same shape as `image`, where background voxels
      are marked as True (1) and others are False (0).
    """
    # Initialize the mask with all False
    mask = np.zeros(image.shape, dtype=bool)

    # Check each dimension for slices that are entirely zero
    for axis in range(3):
        # Sum along the axis; slices that sum to 0 are fully zero
        axis_sum = np.sum(image, axis=axis)
        # Expand the summed axis back to the original shape and compare to 0
        # to mark the entire slice as background
        expanded_axis_sum = np.expand_dims(axis_sum, axis=axis)
        mask |= (expanded_axis_sum == 0)

    # Invert the mask: background voxels marked as True, others as False
    return mask

def main() -> None:
    args = parse_args()
    save_path = Path(args.save_dir)
    mkdir_or_exist(save_path)
    mkdir_or_exist(save_path / "images")
    mkdir_or_exist(save_path / "annotations")
    if args.save_vid:
        mkdir_or_exist(save_path / "videos")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    dataset_name = "MUP"

    # 3D axis to extracts frames
    view = 'axial'

    # Check if the input path ends with .zip
    assert args.path.endswith('.zip'), "MUP should be an archive"

    # Use a temporary directory to unzip the dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Unzipping dataset to temporary directory {temp_dir}")
        shutil.unpack_archive(args.path, temp_dir)
        logging.info("Done")

        file_paths = list(scandir(temp_dir, suffix="nii.gz", recursive=True))
        # print(file_paths)
        img_train = [path for path in file_paths if 'microUS_train' in path]
        annotation_train = [path for path in file_paths if 'expert_annotation_train' in path]
        img_test = [path for path in file_paths if 'microUS_test' in path]
        annotation_test = [path for path in file_paths if 'expert_annotation_test' in path]

        # Function to extract the number from the file name
        def extract_number(file_name: str) -> int:
            match = re.search(r'\d+', file_name)
            return int(match.group()) if match else None

        # Create pairs
        pairs = []
        for us_scan in img_train:
            us_number = extract_number(us_scan)
            for expert_annotation in annotation_train:
                expert_number = extract_number(expert_annotation)
                if us_number == expert_number:
                    pairs.append((us_scan, expert_annotation))
                    break
        for us_scan in img_test:
            us_number = extract_number(us_scan)
            for expert_annotation in annotation_test:
                expert_number = extract_number(expert_annotation)
                if us_number == expert_number:
                    pairs.append((us_scan, expert_annotation))
                    break

        for img_path, mask_path in track_iter_progress(pairs):
            sitk_img = sitk.ReadImage(f"{temp_dir}/{img_path}")
            imgs = sitk.GetArrayFromImage(sitk_img).astype(np.uint8)
            sitk_mask = sitk.ReadImage(f"{temp_dir}/{mask_path}")
            masks = sitk.GetArrayFromImage(sitk_mask).astype(np.uint8)

            # sometime annotation overlap on background
            # fix that by replace with 0, or should we crop?
            bg_mask = create_background_mask(imgs)
            masks[bg_mask] = 0
            # NOTE number id are dupplicated between test and train naming.
            # thus add train and test in patient_id (only last 2 letters)
            patient_idx = mask_path[-12:-10] + mask_path[-9:-7]
            coco = CocoAnnotationObject(
                dataset_name=dataset_name,
                patient_id=patient_idx,
                id_procedure=None,
                save_path=Path(save_path),
                dataset_description=dataset_name,
                dataset_url="https://zenodo.org/records/10475293"
            )
            coco.set_categories([
                {"supercategory": "prostate", "id": 1, "name": "prostate"},
            ])
            for img, mask in zip(imgs, masks):
                img = gray2rgb(img)
                # rotate img / mask for consistency with other dataset
                img = imflip(img, direction='vertical')
                mask = imflip(mask, direction='vertical')
                height, width, _ = img.shape
                coco.add_annotation_from_mask_with_labels(mask)
                image_path = coco.add_image(height, width)
                imwrite(img, image_path)
            coco.dump_to_json()
            if args.save_vid:
                coco.create_video()


if __name__ == '__main__':
    main()
