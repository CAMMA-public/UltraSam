import SimpleITK as sitk
from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np
import json
from skimage import measure
from pycocotools import mask as maskUtils
from skimage.segmentation import flood_fill
from scipy.ndimage import label
from scipy import ndimage
from pathlib import Path
import cv2
import subprocess
from mmcv import imwrite, imread
import uuid


def unpack_rar(rar_file_path, destination_path):
    try:
        subprocess.run(['unrar', 'x', rar_file_path, destination_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while unpacking: {e}")

def fill_mask_labeled(mask):
    # Identify all unique labels in the mask (excluding the background which is assumed to be zero)
    unique_labels = np.unique(mask)
    filled_mask = np.zeros_like(mask)

    for label in unique_labels:
        if label == 0:
            continue
        # Create a binary mask for the current label
        binary_mask = mask == label
        # Fill holes in the binary mask
        filled_binary_mask = ndimage.binary_fill_holes(binary_mask)
        # Assign the label to the filled areas in the output mask
        filled_mask[filled_binary_mask] = label

    return filled_mask

def keep_largest_component(binary_mask):
    """
    Garde la plus grande composante connectée de True dans un masque binaire.
    Les autres composantes sont marquées comme False.

    Parameters:
    - binary_mask: np.array, un masque binaire où True représente les objets.

    Returns:
    - np.array, le masque binaire modifié.
    """
    # Identifier les composantes connectées et le nombre de pixels dans chaque composante
    labeled_array, num_features = label(binary_mask)
    max_label = 0
    max_size = 0

    # Trouver la composante connectée la plus grande
    for i in range(1, num_features + 1):
        component_size = np.sum(labeled_array == i)
        if component_size > max_size:
            max_label = i
            max_size = component_size

    # Créer un nouveau masque où seule la plus grande composante est marquée comme True
    largest_component_mask = (labeled_array == max_label)

    return largest_component_mask


def extract_slices(file_path: str, view: str) -> np.ndarray:
    """
    Extracts sequences of images from a 3D NIfTI file using SimpleITK based on the specified view.
    
    Parameters:
        file_path (str): The path to the `.nii.gz` file.
        view (str): The view to extract ('axial', 'coronal', 'sagittal').
        
    Returns:
        np.ndarray: A 3D numpy array containing the sequences of images.
    """
    # Load the NIfTI file
    sitk_img = sitk.ReadImage(file_path)
    
    # Convert the SimpleITK image to a NumPy array
    data = sitk.GetArrayFromImage(sitk_img)
    
    # Extract based on the specified view
    if view.lower() == 'axial':
        # Assuming the first dimension is the sequence of images
        slices = data
    elif view.lower() == 'coronal':
        # Rearrange dimensions to get coronal view: (z, y, x) -> (y, z, x)
        slices = np.transpose(data, (1, 0, 2))
    elif view.lower() == 'sagittal':
        # Rearrange dimensions to get sagittal view: (z, y, x) -> (x, z, y)
        slices = np.transpose(data, (2, 0, 1))
    else:
        raise ValueError("Invalid view. Choose from 'axial', 'coronal', 'sagittal'.")

    return slices

@dataclass
class CocoAnnotationObject:
    dataset_name: str
    patient_id: str
    id_procedure: str
    save_path: Path  # Base path where images will be saved
    dataset_description: str
    dataset_url: str
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    categories: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    info: Dict[str, Any] = field(init=False)
    _image_id: int = field(default=0, init=False)  # Internal image ID counter

    def __post_init__(self):
        """Initialize dataset info after the object is created."""
        self.info = {
            "description": self.dataset_description,
            "url": self.dataset_url,
        }

    def set_categories(self, categories: List[Dict[str, Any]]):
        """Set the categories for the COCO dataset."""
        self.categories = categories

    def add_image(self, height: int, width: int, label = None) -> Path:
        """Construct image file path and add metadata for an image."""
        parts = [self.dataset_name]
        if self.patient_id is not None:
            parts.append(self.patient_id)
        if self.id_procedure is not None:
            parts.append(self.id_procedure)
        file_name = f"{'__'.join(parts)}__{self._image_id:05}.png"
        full_path = self.save_path / "images" / file_name
        if label is None:
            self.images.append({
                "file_name": file_name,
                "height": height,
                "width": width,
                "id": self._image_id
            })
        else:
            self.images.append({
                "file_name": file_name,
                "height": height,
                "width": width,
                "id": self._image_id,
                "label": label
            })
        self._image_id += 1  # Increment the internal image ID counter
        return full_path

    def add_annotation_from_component(self, labeled_mask: np.ndarray, component_label: int, category_id: int):
        mask_component = labeled_mask == component_label
        labeled_mask_component = np.asarray(mask_component, dtype=np.uint8, order='F')
        mask_rle = maskUtils.encode(labeled_mask_component)

        area = maskUtils.area(mask_rle)
        bbox = maskUtils.toBbox(mask_rle)

        mask_rle['counts'] = mask_rle['counts'].decode('utf-8')

        annotation = {
            "id": int(uuid.uuid4().int),
            "image_id": self._image_id,
            "iscrowd": 0,
            "category_id": category_id,
            "bbox": bbox.tolist(),
            "area": area.tolist(),
            "segmentation": mask_rle
        }

        self.annotations.append(annotation)

    def add_annotations_for_all_components(self, labeled_mask: np.ndarray, category_id: int):
        component_labels = np.unique(labeled_mask)[1:]  # Exclude background

        for component_label in component_labels:
            self.add_annotation_from_component(labeled_mask, component_label, category_id)

    def add_annotation_from_mask_with_labels(self, mask_with_labels: np.ndarray):
        """Create annotations from a mask with multiple labels using the categories information."""
        for category in self.categories:
            category_id = category['id']
            # Extract the binary mask for the current category
            binary_mask = mask_with_labels == category_id

            # Label connected components in the binary mask
            labeled_mask = measure.label(binary_mask, background=0)
            # Use the existing method to add annotations for all components
            self.add_annotations_for_all_components(labeled_mask, category_id)

    def add_annotation_from_polyline(self, segmentation_polylines: list, label: int, image_height: int, image_width: int):
        """
        Add an annotation from a polyline segmentation.

        Args:
        - polyline: List of polyline points as list of lists. Each inner list is one segment of the polyline [[x1, y1, x2, y2, ..., xn, yn]].
        - label: The category ID for the polyline.
        - image_height: The height of the image this annotation is for.
        - image_width: The width of the image this annotation is for.
        """
        # Use pycocotools to convert the segmentation polygon(s) to RLE for area and bbox calculation
        
        mask_rles = maskUtils.frPyObjects(segmentation_polylines, image_height, image_width)
        for mask_rle in mask_rles:
            mask_rle['counts'] = mask_rle['counts'].decode('utf-8')
            area = maskUtils.area(mask_rle)
            bbox = maskUtils.toBbox(mask_rle)

            # Ensure numeric values are in plain Python data types for serialization
            area = float(area)
            bbox = [float(b) for b in bbox]

            annotation = {
                "image_id": self._image_id,  # Ensure this matches your image ID logic
                "iscrowd": 0,
                "category_id": label,
                "bbox": bbox,
                "area": area,
                "segmentation": mask_rle
            }
            self.annotations.append(annotation)


    @classmethod
    def merge_coco_files(cls, coco_files: List[str]) -> 'CocoAnnotationObject':
        merged_coco = cls()
        category_name_to_id = {}
        category_id_counter = 1
        image_id_offset = 0
        annotation_id_offset = 0

        for coco_file in coco_files:
            with open(coco_file, 'r') as file:
                coco_data = json.load(file)
            
            # Merge categories based on name and update annotations' category IDs
            for category in coco_data['categories']:
                cat_name = category['name']
                if cat_name not in category_name_to_id:
                    category_name_to_id[cat_name] = category_id_counter
                    merged_coco.categories.append({
                        'supercategory': category.get('supercategory', ''),
                        'id': category_id_counter,
                        'name': cat_name
                    })
                    category_id_counter += 1
                
            # Update image IDs
            for image in coco_data['images']:
                image['id'] += image_id_offset
                merged_coco.images.append(image)
            image_id_offset = max(img['id'] for img in merged_coco.images) + 1
            
            # Update annotation IDs and their image and category IDs
            for annotation in coco_data['annotations']:
                annotation['id'] += annotation_id_offset
                annotation['image_id'] += image_id_offset
                annotation['category_id'] = category_name_to_id[annotation['category']]
                merged_coco.annotations.append(annotation)
            annotation_id_offset = max(ann['id'] for ann in merged_coco.annotations) + 1

        return merged_coco

    def dump_to_json(self):
        """Export the COCO dataset to a JSON file using logging for messages."""
        coco_format = {
            "info": self.info,
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories
        }

        parts = [self.dataset_name]
        if self.patient_id is not None:
            parts.append(self.patient_id)
        if self.id_procedure is not None:
            parts.append(self.id_procedure)
        file_name = f"{'__'.join(parts)}__coco.json"
        annotations_path = self.save_path / "annotations" / file_name
        with annotations_path.open('w') as f:
            json.dump(coco_format, f)

    def create_video(self, fps: int = 4):
        """Create a video from the COCO dataset with overlaid annotations."""
        # Define a list of distinct colors for each category ID
        output_video_path = str(self.save_path / "videos" / f"{self.dataset_name}_{self.patient_id}_{self.id_procedure}_vid.mp4")
        colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(len(self.categories) + 1)]

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = None

        for image_info in self.images:
            image_path = self.save_path / "images" / image_info['file_name']
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            if video_writer is None:
                height, width, _ = image.shape
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            # Overlay annotations
            for annotation in self.annotations:
                if annotation['image_id'] == image_info['id']:
                    mask = self.decode_rle(annotation['segmentation'])
                    color = colors[annotation['category_id']]
                    image = self.overlay_mask(image, mask, color)

            video_writer.write(image)

        video_writer.release()

    def decode_rle(self, segmentation):
        """Decode RLE segmentation to a binary mask."""
        # Assuming segmentation format is COCO's 'counts' and 'size'
        h, w = segmentation['size']
        mask = maskUtils.decode(segmentation).astype(np.uint8)
        return mask

    def overlay_mask(self, image, mask, color):
        """Overlay a colored mask on the image."""
        colored_mask = np.zeros_like(image)
        for i in range(3):  # Assuming BGR format
            colored_mask[mask == 1, i] = color[i]
        return cv2.addWeighted(image, 1, colored_mask, 0.5, 0)

    def plot_image_with_masks(self, image_id: int):
        """Plot an image with overlaid masks for each annotation using OpenCV."""
        # Find the image info by the given image_id
        colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(len(self.categories) + 1)]
        image_info = next((img for img in self.images if img['id'] == image_id), None)
        if not image_info:
            print(f"Image with ID {image_id} not found.")
            return

        # Load the image from its path
        image_path = self.save_path / "images" / image_info['file_name']
        image = imread(str(image_path))
        if image is None:
            print(f"Failed to load image from path: {image_path}")
            return

        # Overlay annotations on the image
        for annotation in self.annotations:
            if annotation['image_id'] == image_id:
                mask = self.decode_rle(annotation['segmentation'])
                color = colors[annotation['category_id']]
                image = self.overlay_mask(image, mask, color)
        imwrite(image, self.save_path / "vizualisation" / image_info['file_name'])