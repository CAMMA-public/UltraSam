from mmdet.registry import TRANSFORMS
from mmdet.datasets.transforms import PackDetInputs


@TRANSFORMS.register_module()
class PackPointDetInputs(PackDetInputs):

    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks',
        'points': 'points',
        'boxes': 'boxes',
        'prompt_types': 'prompt_types'
    }
