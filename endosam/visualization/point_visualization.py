from typing import Dict, List, Optional, Tuple, Union

import cv2
import mmcv
import numpy as np
import torch
from mmengine.dist import master_only
from mmengine.structures import InstanceData, PixelData, LabelData
from mmengine.visualization import Visualizer

from mmdet.evaluation import INSTANCE_OFFSET
from mmdet.registry import VISUALIZERS
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks, PolygonMasks, bitmap_to_polygon
from mmdet.visualization.palette import _get_adaptive_scales, get_palette, jitter_color
from mmdet.visualization import DetLocalVisualizer


@VISUALIZERS.register_module()
class PointVisualizer(DetLocalVisualizer):
    def _draw_instances(self, image: np.ndarray, instances: ['InstanceData'],
                        classes: Optional[List[str]],
                        palette: Optional[List[tuple]]) -> np.ndarray:
        """Draw instances of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            classes (List[str], optional): Category information.
            palette (List[tuple], optional): Palette information
                corresponding to the category.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        self.set_image(image)

        if 'bboxes' in instances and instances.bboxes.sum() > 0:
            bboxes = instances.bboxes
            labels = instances.labels

            max_label = int(max(labels) if len(labels) > 0 else 0)
            text_palette = get_palette(self.text_color, max_label + 1)
            text_colors = [text_palette[label] for label in labels]

            bbox_color = palette if self.bbox_color is None \
                else self.bbox_color
            bbox_palette = get_palette(bbox_color, max_label + 1)
            colors = [bbox_palette[label] for label in labels]
            # self.draw_bboxes(
            #     bboxes,
            #     edge_colors=colors,
            #     alpha=self.alpha,
            #     line_widths=self.line_width * 2 )

            positions = bboxes[:, :2] + self.line_width
            areas = (bboxes[:, 3] - bboxes[:, 1]) * (
                bboxes[:, 2] - bboxes[:, 0])
            scales = _get_adaptive_scales(areas)

            for i, (pos, label) in enumerate(zip(positions, labels)):
                if 'label_names' in instances:
                    label_text = instances.label_names[i]
                else:
                    label_text = classes[
                        label] if classes is not None else f'class {label}'
                if 'scores' in instances:
                    score = round(float(instances.scores[i]) * 100, 1)
                    label_text += f': {score}'

                # self.draw_texts(
                #     label_text,
                #     pos,
                #     colors=text_colors[i],
                #     font_sizes=int(13 * 2 * scales[i]),
                #     bboxes=[{
                #         'facecolor': 'black',
                #         'alpha': 0.8,
                #         'pad': 0.7,
                #         'edgecolor': 'none'
                #     }])

        if 'masks' in instances:
            labels = instances.labels
            masks = instances.masks
            if isinstance(masks, torch.Tensor):
                masks = masks.numpy()
            elif isinstance(masks, (PolygonMasks, BitmapMasks)):
                masks = masks.to_ndarray()

            masks = masks.astype(bool)

            max_label = int(max(labels) if len(labels) > 0 else 0)
            mask_color = palette if self.mask_color is None \
                else self.mask_color
            mask_palette = get_palette(mask_color, max_label + 1)
            # colors = [jitter_color(mask_palette[label]) for label in labels]
            colors = [mask_palette[label] for label in labels]
            text_palette = get_palette(self.text_color, max_label + 1)
            text_colors = [text_palette[label] for label in labels]

            polygons = []
            for i, mask in enumerate(masks):
                contours, _ = bitmap_to_polygon(mask)
                polygons.extend(contours)
            self.draw_polygons(polygons, edge_colors='w', alpha=self.alpha)
            self.draw_binary_masks(masks, colors=colors, alphas=self.alpha)

            #if len(labels) > 0 and \
            #        ('bboxes' not in instances or
            #         instances.bboxes.sum() == 0):
            #    # instances.bboxes.sum()==0 represent dummy bboxes.
            #    # A typical example of SOLO does not exist bbox branch.
            #    areas = []
            #    positions = []
            #    for mask in masks:
            #        _, _, stats, centroids = cv2.connectedComponentsWithStats(
            #            mask.astype(np.uint8), connectivity=8)
            #        if stats.shape[0] > 1:
            #            largest_id = np.argmax(stats[1:, -1]) + 1
            #            positions.append(centroids[largest_id])
            #            areas.append(stats[largest_id, -1])
            #    areas = np.stack(areas, axis=0)
            #    scales = _get_adaptive_scales(areas)

            #    for i, (pos, label) in enumerate(zip(positions, labels)):
            #        if 'label_names' in instances:
            #            label_text = instances.label_names[i]
            #        else:
            #            label_text = classes[
            #                label] if classes is not None else f'class {label}'
            #        if 'scores' in instances:
            #            score = round(float(instances.scores[i]) * 100, 1)
            #            label_text += f': {score}'

            #        self.draw_texts(
            #            label_text,
            #            pos,
            #            colors=text_colors[i],
            #            font_sizes=int(13 * scales[i]),
            #            horizontal_alignments='center',
            #            bboxes=[{
            #                'facecolor': 'black',
            #                'alpha': 0.8,
            #                'pad': 0.7,
            #                'edgecolor': 'none'
            #            }])
        #super()._draw_instances(image, instances, classes, palette)

        if 'bboxes' in instances and 'scores' not in instances:
            bboxes = instances.bboxes
            points = (bboxes[:, :2] + bboxes[:, 2:]) / 2
            labels = instances.labels
            # print(instances)

            max_label = int(max(labels) if len(labels) > 0 else 0)
            # text_palette = get_palette(self.text_color, max_label + 1)
            # text_colors = [text_palette[label] for label in labels]

            bbox_color = palette if self.bbox_color is None \
                else self.bbox_color
            bbox_palette = get_palette(bbox_color, max_label + 1)
            colors = [bbox_palette[label] for label in labels]
            # plot points
            for i, pts in enumerate(points):
                # outline
                img_wh = torch.as_tensor(image.shape[:2][::-1])  # Ensure it's a tensor
                pts = torch.as_tensor(pts)  # Ensure it's a tensor
                self.draw_points(
                    pts,
                    colors='w',
                    sizes=np.array([[40  * 2 ]], dtype=np.float32)
                )
                self.draw_points(
                    pts,
                    # colors=[colors[i]],
                    colors='tab:blue',
                    sizes=np.array([[35 * 2 ]], dtype=np.float32)
                )

        return self.get_image()
