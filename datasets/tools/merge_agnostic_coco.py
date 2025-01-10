import json
import os
import uuid
from pathlib import Path


def merge_agnostic_coco(dataset_dir, mode="train", output_path="merged_coco.json"):
    merged_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "object", "supercategory": "none"}]  # Assuming agnostic has one category
    }

    # Go through each dataset directory
    for dataset in os.listdir(dataset_dir):
        dataset_path = os.path.join(dataset_dir, dataset)
        annotations_dir = os.path.join(dataset_path, "annotations")
        if not os.path.exists(annotations_dir):
            print(f"{annotations_dir} does not exist...")
            continue

        agnostic_coco_file = f"{mode}.agnostic.{dataset}__coco.json"
        coco_path = os.path.join(annotations_dir, agnostic_coco_file)

        if not os.path.exists(coco_path):
            print(f"{coco_path} does not exist...")
            continue

        # Load the COCO annotations
        with open(coco_path, "r") as f:
            coco_data = json.load(f)

        # Add images with updated IDs and updated file paths
        for image in coco_data['images']:
            old_image_id = image['id']  # Store the old ID
            new_image_id = uuid.uuid4().int  # Generate a new unique image ID
            image['id'] = new_image_id

            # Update the file path to include the dataset name and images directory
            image['file_name'] = f"{dataset}/images/{image['file_name']}"

            merged_data['images'].append(image)

            # Update annotations with new image ID and assign unique annotation IDs
            for annotation in coco_data['annotations']:
                if annotation['image_id'] == old_image_id:
                    annotation['image_id'] = new_image_id
                    annotation['id'] = uuid.uuid4().int
                    merged_data['annotations'].append(annotation)

    # Save the merged COCO annotations
    with open(output_path, "w") as f:
        json.dump(merged_data, f, indent=4)

    print(f"Merged COCO file saved to {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python merge_agnostic_coco.py <dataset_dir> <output_path> [--mode <train/val/test>]")
    else:
        dataset_dir = sys.argv[1]
        output_path = sys.argv[2]
        mode = sys.argv[4] if len(sys.argv) > 3 and sys.argv[3] == "--mode" else "train"
        merge_agnostic_coco(dataset_dir, mode, output_path)
