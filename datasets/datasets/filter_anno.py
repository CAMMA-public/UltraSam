import json
from collections import defaultdict

# Load the JSON data
file_path = '/home/ameyer/serveurs/unistra_data/camma_data/UltraSAM/MMOTU_2d/annotations/test.MMOTU_2d__coco.json'  # Replace with your actual file path
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract the annotations
annotations = data.get("annotations", [])

# Group annotations by image_id
annotations_by_image = defaultdict(list)
for annotation in annotations:
    image_id = annotation['image_id']
    annotations_by_image[image_id].append(annotation)

# Function to filter out small annotations
def filter_annotations(annotations):
    if not annotations:
        return annotations
    
    # Find the largest annotation in this image
    max_area = max(annotations, key=lambda x: x['area'])['area']
    
    # Keep annotations that are at least 5% of the largest area
    filtered_annotations = [ann for ann in annotations if ann['area'] >= 0.95 * max_area]
    
    return filtered_annotations

# Apply the filter function to each image's annotations
filtered_annotations_by_image = {image_id: filter_annotations(anns) for image_id, anns in annotations_by_image.items()}

# Flatten the result back into a list of annotations
filtered_annotations = [ann for anns in filtered_annotations_by_image.values() for ann in anns]

len_before = len(annotations)
len_after = len(filtered_annotations)

print(len_before, len_after)

# Prepare the updated COCO data structure with filtered annotations
updated_coco_data = data.copy()
updated_coco_data['annotations'] = filtered_annotations

min_area = 0
for ann in filtered_annotations:
    if ann['area'] > min_area:
        min_anno = ann
print("smallest annotation:\n")
print(min_anno)

# Specify the output path
output_path = '/home/ameyer/serveurs/unistra_data/camma_data/UltraSAM/MMOTU_2d/annotations/test.MMOTU_2d__coco.json'  # Replace with your desired output path

# Save the updated COCO JSON to the specified path
with open(output_path, 'w') as outfile:
    json.dump(updated_coco_data, outfile)

print(f"Updated COCO dataset saved to {output_path}")
