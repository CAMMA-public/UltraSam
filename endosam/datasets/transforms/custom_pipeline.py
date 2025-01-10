from mmcv.transforms import BaseTransform
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmdet.registry import TRANSFORMS
import numpy as np
import random
import torch
from typing import List, Optional, Sequence, Tuple, Union, Dict
from enum import Enum


# Define the Enum for prompt types
class PromptType(Enum):
    POINT = 0
    BOX = 1

def get_random_number(numbers: List[int]) -> Union[int, ValueError]:
    """
    Returns a random number from a list with either one or two integer elements.
    If the list has one element, returns that element.
    If the list has two elements, returns a random integer between those two elements (inclusive).

    Args:
    numbers (List[int]): A list of integers containing either one or two elements.

    Returns:
    Union[int, ValueError]: The number or a ValueError if the list does not meet the requirements.

    Raises:
    ValueError: If the list has neither one nor two elements.
    """
    if len(numbers) == 1:
        return numbers[0]
    elif len(numbers) == 2:
        return random.randint(numbers[0], numbers[1])
    else:
        raise ValueError("List must have one or two elements.")

@TRANSFORMS.register_module()
class GetPointFromMask(BaseTransform):
    """
    Class to get a random index of a pixel inside the mask and store the coordinate of the chosen point.
    """
    def __init__(self, number_of_points: Union[int, List[int]] = 3, normalize: bool = True,
            test: bool = False, get_center_point: bool = False):
        """
        Initializes the GetPointFromMask object.

        Args:
            number_of_points (Union[int, List[int]]): The number of points to be generated. Can be a single integer or a list.
            normalize (bool): Whether to normalize the coordinates of the point.
            test (bool): Whether the class is in test mode.
        """
        self.number_of_points = number_of_points
        self.normalize = normalize
        self.test = test
        self.get_center_point = get_center_point

    def _getPointMask(self, results) -> List[Tuple[np.ndarray, np.ndarray]]:
        mask_arrays = results["gt_masks"].masks
        img_key = "img_shape"
        if self.normalize:
            img_height, img_width = results[img_key]
        else:
            img_height, img_width = 1, 1
        if self.test:
            x_scale, y_scale = results["scale_factor"]

        points_list = []

        for mask_array in mask_arrays:
            y_indices, x_indices = np.nonzero(mask_array)
            n_points = get_random_number(self.number_of_points)

            selected_indices = np.random.choice(len(x_indices), size=n_points, replace=True)
            jitter = 0 if self.test else np.random.random(n_points) * (1 - np.finfo(float).eps)

            if self.get_center_point:
                if isinstance(self.number_of_points, int) and self.number_of_points > 1:
                    raise NotImplementedError
                elif isinstance(self.number_of_points, list):
                    if max(self.number_of_points) > 1:
                        raise NotImplementedError

                indices = np.argwhere(mask_array)

                # Calculate the mean of the indices along each axis
                y_points, x_points = np.split(indices.mean(axis=0), 2)
                if self.test:
                    x_points = (x_points + jitter) * x_scale / img_width + 0.5
                    y_points = (y_points + jitter) * y_scale / img_height + 0.5
                else:
                    x_points = (x_points + jitter) / img_width
                    y_points = (y_points + jitter) / img_height

            else:
                if self.test:
                    x_points = (x_indices[selected_indices] + jitter) * x_scale / img_width + 0.5
                    y_points = (y_indices[selected_indices] + jitter) * y_scale / img_height + 0.5
                else:
                    x_points = (x_indices[selected_indices] + jitter) / img_width
                    y_points = (y_indices[selected_indices] + jitter) / img_height

            index_candidat = np.stack((x_points, y_points), axis=-1)
            points_list.append(index_candidat)

        return points_list

    def transform(self, results):
        points = self._getPointMask(results)
        results["points"] = np.array(points)

        return results
    
@TRANSFORMS.register_module()
class GetPointFromBox(BaseTransform):
    """
    Class to get random points inside bounding boxes and store the coordinates of the chosen points.
    """
    def __init__(self, number_of_points: Union[int, List[int]] = 3, normalize: bool = True,
                 test: bool = False, get_center_point: bool = False):
        """
        Initializes the GetPointFromBox object.

        Args:
            number_of_points (Union[int, List[int]]): The number of points to be generated. Can be a single integer or a list.
            normalize (bool): Whether to normalize the coordinates of the point.
            test (bool): Whether the class is in test mode.
            get_center_point (bool): Whether to get the center point of the bounding box.
        """
        self.number_of_points = number_of_points
        self.normalize = normalize
        self.test = test
        self.get_center_point = get_center_point

    def _getPointFromBox(self, results) -> List[Tuple[np.ndarray, np.ndarray]]:
        bbox_arrays = results["gt_bboxes"].tensor
        img_key = "img_shape"
        if self.normalize:
            img_height, img_width = results[img_key]
        else:
            img_height, img_width = 1, 1
        if self.test:
            x_scale, y_scale = results["scale_factor"]

        points_list = []

        for bbox in bbox_arrays:
            xmin, ymin, xmax, ymax = bbox
            if self.get_center_point:
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                x_points = np.array([x_center])
                y_points = np.array([y_center])
            else:
                n_points = get_random_number(self.number_of_points)
                x_points = np.random.uniform(xmin, xmax, size=n_points)
                y_points = np.random.uniform(ymin, ymax, size=n_points)

            if self.test:
                x_points = (x_points * x_scale) / img_width + 0.5
                y_points = (y_points * y_scale) / img_height + 0.5
            else:
                x_points = x_points / img_width
                y_points = y_points / img_height

            index_candidat = np.stack((x_points, y_points), axis=-1)
            points_list.append(index_candidat)

        return points_list

    def transform(self, results):
        points = self._getPointFromBox(results)
        results["points"] = np.array(points)

        return results



@TRANSFORMS.register_module()
class GetPointBox(BaseTransform):
    def __init__(self, normalize: bool = True, max_jitter: float = 0.05, test: bool = False):
        """
        Initializes the GetPointBox object.

        Args:
            max_jitter (float): Max jitter to move each box corner as a fraction of box dimensions.
            normalize (bool): Whether to normalize the coordinates of the point.
            test (bool): Whether the class is in test mode.
        """
        self.normalize = normalize
        self.max_jitter = max_jitter
        self.test = test

    def _apply_jitter(self, x_points: torch.Tensor, y_points: torch.Tensor, box_width: torch.Tensor, box_height: torch.Tensor):
        """
        Apply jitter to the box corners.

        Args:
            x_points (torch.Tensor): Tensor containing xmin and xmax.
            y_points (torch.Tensor): Tensor containing ymin and ymax.
            box_width (torch.Tensor): Width of the bounding box.
            box_height (torch.Tensor): Height of the bounding box.

        Returns:
            (torch.Tensor, torch.Tensor): Jittered x_points and y_points.
        """
        # Generate different jitter for each corner
        jitter_x = torch.rand(2) * 2 * self.max_jitter - self.max_jitter  # Uniform random jitter between -max_jitter and +max_jitter
        jitter_y = torch.rand(2) * 2 * self.max_jitter - self.max_jitter

        # Apply jitter to each corner scaled by box width/height
        x_points += jitter_x * box_width
        y_points += jitter_y * box_height

        return x_points, y_points

    def _getPointBox(self, results) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        bbox_arrays = results["gt_bboxes"].tensor
        img_key = "img_shape"
        if self.normalize:
            img_height, img_width = results[img_key]
        else:
            img_height, img_width = 1, 1
        if self.test:
            x_scale, y_scale = results["scale_factor"]

        points_list = []

        for bbox in bbox_arrays:
            xmin, ymin, xmax, ymax = bbox
            x_points = torch.tensor([xmin, xmax], dtype=bbox.dtype)
            y_points = torch.tensor([ymin, ymax], dtype=bbox.dtype)

            box_width = xmax - xmin
            box_height = ymax - ymin

            # Apply jitter to the corners
            x_points, y_points = self._apply_jitter(x_points, y_points, box_width, box_height)

            if self.test:
                x_points = x_points * x_scale / img_width + 0.5
                y_points = y_points * y_scale / img_height + 0.5
            else:
                x_points = x_points / img_width + 0.5
                y_points = y_points / img_height + 0.5

            index_candidat = torch.stack((x_points, y_points), dim=-1)
            points_list.append(index_candidat)

        return points_list

    def transform(self, results):
        boxes = self._getPointBox(results)
        results["boxes"] = torch.stack(boxes)

        return results


@TRANSFORMS.register_module()
class GetPromptType(BaseTransform):
    def __init__(self, prompt_type=[PromptType.POINT, PromptType.BOX],
                 prompt_probabilities=[0.5, 0.5]):
        """
        Initializes the GetPromptType object.

        Args:
            prompt_type (list): List of possible prompt types to randomly select from.
            prompt_probabilities (list): List of probabilities corresponding to each prompt type.
        """
        self.prompt_type = prompt_type
        self.prompt_probabilities = prompt_probabilities

        # Ensure that probabilities are valid
        assert len(self.prompt_type) == len(self.prompt_probabilities), \
            "The number of prompt types must match the number of probabilities."
        assert np.isclose(sum(self.prompt_probabilities), 1.0), \
            "Probabilities must sum to 1."

    def transform(self, results):
        n_boxes, _, _ = results["boxes"].shape

        # Create a numpy array with sampling based on the given probabilities
        prompt_array = np.random.choice(
            [pt.value for pt in self.prompt_type],  # List of enum values
            size=n_boxes,
            p=self.prompt_probabilities  # Probabilities for each type
        )

        # Add the array to the results dictionary
        results["prompt_types"] = prompt_array

        return results
