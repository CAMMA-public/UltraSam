import logging
import argparse
import tempfile
from pathlib import Path
import shutil
from utils import extract_slices, CocoAnnotationObject
from mmengine.utils import track_iter_progress, mkdir_or_exist, scandir
from mmcv import imshow, gray2bgr, gray2rgb, imwrite
import numpy as np

dataset_name = "CAMUS"

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert CAMUS to frames & COCO style annotations')
    parser.add_argument(
        '--path',
        type=str,
        help='dataset path',
        default='/DATA/CAMUS_public.zip')
    parser.add_argument(
        '--save-dir',
        type=str,
        help='the dir to save dataset',
        default=f"/media/ameyer/Data4/ULTRASam/datasets/{dataset_name}")
    parser.add_argument(
        '--save-vid',
        action='store_false',
        help='save img vizualisation')
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


def main() -> None:
    args = parse_args()
    save_path = Path(args.save_dir)
    mkdir_or_exist(save_path)
    mkdir_or_exist(save_path / "images")
    mkdir_or_exist(save_path / "annotations")
    if args.save_vid:
        mkdir_or_exist(save_path / "videos")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    dataset_name = "CAMUS"

    # 3D axis to extracts frames
    view = 'axial'

    # Check if the input path ends with .zip
    assert args.path.endswith('.zip'), "CAMUS should be an archive"

    # Use a temporary directory to unzip the dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Unzipping dataset to temporary directory {temp_dir}")
        shutil.unpack_archive(args.path, temp_dir)
        logging.info("Done")

        file_paths = list(scandir(temp_dir, suffix="nii.gz", recursive=True))

        required_files = ['2CH_half_sequence.nii.gz',
                          '2CH_half_sequence_gt.nii.gz',
                          '4CH_half_sequence.nii.gz',
                          '4CH_half_sequence_gt.nii.gz']
        patients = {}

        for path in file_paths:
            patient_id = path.split('/')[-2]

            # collect desired files per patient
            if any(path.endswith(req_file) for req_file in required_files):
                if patient_id not in patients:
                    patients[patient_id] = []
                patients[patient_id].append(path)

        # Asserting each patient has all required files
        for patient_id, files in patients.items():
            for req_file in required_files:
                assert any(req_file in file.split('/')[-1] for file in files), f"{patient_id} is missing {req_file}"

        patient_ids = [patient_id for patient_id in patients.keys()]
        for patient_id in track_iter_progress(patient_ids):
            # Sort the files for the patient based on the order in required_files
            sorted_files = sorted(patients[patient_id], key=lambda x: required_files.index(x.split('/')[-1][12:]))
            CH2_images, CH2_masks, CH4_images, CH4_masks = sorted_files

            # extract img & masks
            CH2_images_slices = extract_slices(f"{temp_dir}/{CH2_images}", view).astype(np.uint8)
            CH2_masks_slices = extract_slices(f"{temp_dir}/{CH2_masks}", view).astype(np.uint8)
            CH4_images_slices = extract_slices(f"{temp_dir}/{CH4_images}", view).astype(np.uint8)
            CH4_masks_slices = extract_slices(f"{temp_dir}/{CH4_masks}", view).astype(np.uint8)

            # masking to remove segmentation mask error outside of US FOV
            invalid_mask_CH2_images_slices = CH2_images_slices == 0
            invalid_mask_CH4_images_slices = CH4_images_slices == 0
            CH2_masks_slices[invalid_mask_CH2_images_slices] = 0
            CH4_masks_slices[invalid_mask_CH4_images_slices] = 0

            tmp = ([CH2_images_slices, CH4_images_slices], [CH2_masks_slices, CH4_masks_slices], ["CH2", "CH4"])
            for procedures, masks, id_procedure in zip(*tmp):
                coco = CocoAnnotationObject(
                    dataset_name=dataset_name,
                    patient_id=patient_id[-4:],
                    id_procedure=id_procedure,
                    save_path=Path(save_path),
                    dataset_description=dataset_name,
                    dataset_url="https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8g"
                )
                coco.set_categories([
                    {"supercategory": "endocardium", "id": 1, "name": "endocardium"},
                    {"supercategory": "epicardium", "id": 2, "name": "epicardium"},
                    {"supercategory": "atrium_wall", "id": 3, "name": "atrium_wall"},
                ])
                for procedure, mask in zip(procedures, masks):
                    procedure = gray2rgb(procedure)
                    height, width, _ = procedure.shape
                    coco.add_annotation_from_mask_with_labels(mask)
                    image_path = coco.add_image(height, width)
                    imwrite(procedure, image_path)

                coco.dump_to_json()
                if args.save_vid:
                    coco.create_video()


if __name__ == '__main__':
    main()
