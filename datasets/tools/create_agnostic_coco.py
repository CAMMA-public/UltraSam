import json
import os
import uuid

def create_agnostic_annotations(dataset_dir, mode="val.agnostic"):
    # Go through each dataset directory
    for dataset in os.listdir(dataset_dir):
        dataset_path = os.path.join(dataset_dir, dataset)
        annotations_dir = os.path.join(dataset_path, "annotations")
        if not os.path.exists(annotations_dir):
            print(f"{annotations_dir} does not exist...")
            continue

        coco_file = f"{mode}.{dataset}__coco.json"
        coco_path = os.path.join(annotations_dir, coco_file)

        if not os.path.exists(coco_path):
            print(f"{coco_path} does not exist...")
            continue

        # Load the COCO annotations
        with open(coco_path, "r") as f:
            coco_data = json.load(f)

        # Create a single generic category
        agnostic_category = {"id": 1, "name": "object", "supercategory": "none"}

        # Update the category list to contain only the generic category
        coco_data["categories"] = [agnostic_category]

        # Update each annotation to refer to the generic category (category_id=1)
        for annotation in coco_data["annotations"]:
            annotation["category_id"] = 1
            annotation['id'] = int(uuid.uuid4().int)

        # Define the output file name
        # agnostic_coco_file = f"{mode}.agnostic.{dataset}__coco.json"
        agnostic_coco_file = f"{mode}.{dataset}__coco.json"
        agnostic_coco_path = os.path.join(annotations_dir, agnostic_coco_file)

        # Save the agnostic annotations
        with open(agnostic_coco_path, "w") as f:
            json.dump(coco_data, f, indent=4)

        print(f"Created class-agnostic annotations for {dataset} at {agnostic_coco_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python create_agnostic_annotations.py <dataset_dir> [--mode <train/val/test>]")
    else:
        dataset_dir = sys.argv[1]
        mode = sys.argv[3] if len(sys.argv) > 2 and sys.argv[2] == "--mode" else "train"
        create_agnostic_annotations(dataset_dir, mode)
