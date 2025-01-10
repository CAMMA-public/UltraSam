import torch
from mmengine.structures import InstanceData

from mmdet.registry import TASK_UTILS
from mmdet.models.task_modules.assigners.assign_result import AssignResult
from mmdet.models.task_modules.assigners.base_assigner import BaseAssigner


@TASK_UTILS.register_module()
class SAMassigner(BaseAssigner):
    def assign(self,
               gt_instances: InstanceData,
               num_multi_mask: int = 1,
               **kwargs) -> AssignResult:
        """Assign boxes to the already know GT.
        NOTE we return 1 match per instance (trivial assignent)
        in case of multiple mask_token, should compute IOU matching outside
        NO SHOULD RETURN 1 MATCH PER PRED MASK
        Then backpropagate only the smallest lost per instance

        can keep like that and use repeat_interleave?
        """
        # gt_masks = gt_instances.masks
        gt_labels = gt_instances.labels
        device = gt_labels.device

        if gt_instances.points.shape[0] == 0:  # Empty points
            return AssignResult(
                num_gts=0,
                gt_inds=torch.tensor([], dtype=torch.long, device=device),
                max_overlaps=None,
                labels=torch.tensor([], dtype=torch.long, device=device)
            )

        num_gts = len(gt_labels)
        assigned_gt_inds = torch.arange(
            num_gts,
            dtype=torch.long, device=device)

        return AssignResult(
            num_gts=num_gts * num_multi_mask,
            gt_inds=assigned_gt_inds.repeat_interleave(num_multi_mask),
            max_overlaps=None,
            labels=gt_labels.repeat_interleave(num_multi_mask)
        )
