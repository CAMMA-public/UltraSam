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
from mmcv import imshow, gray2bgr, gray2rgb, rgb2gray, bgr2gray, imwrite, imread, imrotate, imflip, VideoReader
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
import pandas as pd
from shapely.geometry import LineString, Polygon, Point

dataset_name = "echonetpediatric"

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


def create_segmentation_masks(df: pd.DataFrame, data_path: str) -> dict:
    segmentation_masks = {}

    # Group DataFrame by video name
    grouped = df.groupby('FileName')

    for video_name, group_df in grouped:
        video_masks = {}

        # Group DataFrame by frame
        frame_grouped = group_df.groupby('Frame')

        for frame_id, frame_df in frame_grouped:
            if frame_id == "No Systolic" or frame_id == "No Diastolic":
                continue
            video_path = os.path.join(data_path, "Videos", video_name)
            if not os.path.exists(video_path):
                print(f"Skipping {video_name}: video file not found.")
                continue
            vid = VideoReader(f"{data_path}/Videos/{video_name}")
            width, height = vid.width, vid.height
            # Extract lines for the frame
            x1 = frame_df['X'].tolist()
            y1 = frame_df['Y'].tolist()

            # Create LineString objects for each line
            points = [(x1[i], y1[i]) for i in range(len(x1))]
            # Ensure the points form a closed polygon
            if points[0] != points[-1]:
                points.append(points[0])

            # Convert points to a numpy array of shape (n, 1, 2)
            poly_points = np.array([points], dtype=np.int32)

            # Create an empty mask and draw the polygon
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, poly_points, 1)  # 1: fill color

            # Store the binary mask in the dictionary
            video_masks[frame_id] = mask.astype(np.uint8)

        # Store video masks in the segmentation_masks dictionary
        segmentation_masks[video_name] = video_masks

    return segmentation_masks


def main() -> None:
    args = parse_args()
    save_path = Path(args.save_dir)
    mkdir_or_exist(save_path)
    mkdir_or_exist(save_path / "images")
    mkdir_or_exist(save_path / "annotations")
    if args.save_viz:
        mkdir_or_exist(save_path / "vizualisation")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    for view in ['A4C', 'PSAX']:
        data_path = f"{args.path}/pediatric_echo_avi/pediatric_echo_avi/{view}"

        df = pd.read_csv(f"{data_path}/VolumeTracings.csv")
        segmentation_masks = create_segmentation_masks(df, data_path)

        for video_name in segmentation_masks.keys():
            coco = CocoAnnotationObject(
                dataset_name=dataset_name,
                patient_id=video_name.split(".")[0],
                id_procedure=view,
                save_path=Path(save_path),
                dataset_description=dataset_name,
                dataset_url="https://echonet.github.io/pediatric"
            )
            coco.set_categories([
                {"supercategory": "left_ventricle", "id": 1, "name": "left_ventricle"},
            ])
            for frame_id in segmentation_masks[video_name]:
                print(video_name, frame_id)

                vid = VideoReader(f"{data_path}/Videos/{video_name}")
                print(vid.width, vid.height, vid.resolution, vid.fps, len(vid))
                img = vid[int(frame_id)-1]
                if img is None:
                    continue
                mask = segmentation_masks[video_name][frame_id]
                # imshow(vid[frame_id])
                # imshow(segmentation_masks[video_name][frame_id]*255)

                height, width, _ = img.shape
                coco.add_annotation_from_mask_with_labels(mask)
                image_path = coco.add_image(height, width)
                imwrite(img, image_path)
                if args.save_viz:
                    coco.plot_image_with_masks(coco._image_id-1)

            coco.dump_to_json()


if __name__ == '__main__':
    main()
