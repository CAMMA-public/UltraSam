import json
import os
import random
import argparse
from pathlib import Path
from shutil import copyfile
from collections import defaultdict

def category_check(annotations):
    category_count = defaultdict(int)
    for ann in annotations:
        category_count[ann["category_id"]] += 1
    return category_count

def categories_present_in_both(train_category_count, val_category_count, all_categories):
    """Check if each category is present in both training and validation sets."""
    for category in all_categories:
        if train_category_count[category['id']] == 0 or val_category_count[category['id']] == 0:
            # print(f"{category['id']} train: {train_category_count[category['id']]}, val: {val_category_count[category['id']]}")
            return False
    return True

def split_annotations(dataset_dir, train_percent=0.95):
    # Go through each dataset directory
    for dataset in os.listdir(dataset_dir):
        dataset_path = os.path.join(dataset_dir, dataset)
        annotations_dir = os.path.join(dataset_path, "annotations")
        if not os.path.exists(annotations_dir):
            print(f"{annotations_dir} does not exist...")
            continue

        coco_file = f"{dataset}__coco.json"
        coco_path = os.path.join(annotations_dir, coco_file)

        if not os.path.exists(coco_path):
            print(f"{coco_path} does not exist...")
            continue

        # Load the COCO annotations
        with open(coco_path, "r") as f:
            coco_data = json.load(f)

        print(f"\n=========  {dataset}")
        # Extract image and annotation information
        images = coco_data["images"]
        annotations = coco_data["annotations"]
        categories = coco_data["categories"]
        train_annotations = []
        val_annotations = []
        max_retries = 10000  # Maximum number of reshuffles to try

        for _ in range(max_retries):
            # Shuffle images
            random.shuffle(images)
            split_idx = int(len(images) * train_percent)

            train_images = images[:split_idx]
            val_images = images[split_idx:]

            # Get image ids for train and validation
            train_ids = set([img["id"] for img in train_images])
            val_ids = set([img["id"] for img in val_images])

            # Split annotations based on image ids
            train_annotations = [ann for ann in annotations if ann["image_id"] in train_ids]
            val_annotations = [ann for ann in annotations if ann["image_id"] in val_ids]

            # Check category presence
            train_category_count = category_check(train_annotations)
            val_category_count = category_check(val_annotations)

            if categories_present_in_both(train_category_count,
                                          val_category_count,
                                          categories):
                break
        else:
            print(f"Warning: Could not ensure category balance after {max_retries} retries for {dataset}")

        # Create train and val coco.json files
        train_coco = {
            "images": train_images,
            "annotations": train_annotations,
            "categories": coco_data["categories"],
        }
        val_coco = {
            "images": val_images,
            "annotations": val_annotations,
            "categories": coco_data["categories"],
        }

        train_file = os.path.join(annotations_dir, f"train.{dataset}__coco.json")
        val_file = os.path.join(annotations_dir, f"val.{dataset}__coco.json")

        # Save the new coco files
        with open(train_file, "w") as f:
            json.dump(train_coco, f, indent=4)
        with open(val_file, "w") as f:
            json.dump(val_coco, f, indent=4)

        print(f"Created train and val files for {dataset}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset annotations into train and val COCO files")
    parser.add_argument("dataset_dir", type=str, help="Path to the dataset directory")
    parser.add_argument("--train_percent", type=float, default=0.95, help="Percentage of training data (default: 95%)")

    args = parser.parse_args()
    split_annotations(args.dataset_dir, train_percent=args.train_percent)
