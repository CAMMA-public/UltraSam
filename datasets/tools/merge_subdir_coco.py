import os
import json
import uuid


def merge_coco_files(dir_path):
    merged_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    category_set = set()

    # Loop through all json files in the directory
    for file_name in os.listdir(dir_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(dir_path, file_name)
            with open(file_path, 'r') as f:
                coco_data = json.load(f)

            # Ensure all categories are consistent
            for category in coco_data['categories']:
                category_tuple = (category['id'], category['name'])
                if category_tuple not in category_set:
                    category_set.add(category_tuple)
                    merged_data['categories'].append(category)

            # Add images with updated IDs
            for image in coco_data['images']:
                old_image_id = image['id']  # Store the old ID
                new_image_id = uuid.uuid4().int
                image['id'] = new_image_id
                merged_data['images'].append(image)

                # Update annotations with new image ID and unique annotation ID
                for annotation in coco_data['annotations']:
                    if annotation['image_id'] == old_image_id:
                        annotation['image_id'] = new_image_id
                        merged_data['annotations'].append(annotation)


    # Save the merged COCO file
    output_file = os.path.join(dir_path, f"{os.path.basename(os.path.dirname(dir_path))}__coco.json")
    with open(output_file, 'w') as out_file:
        json.dump(merged_data, out_file)

    print(f"COCO files merged into: {output_file}")


dir_path = 'path_to_ultrasound-nerve-segmentation/annotations'
merge_coco_files(dir_path)
