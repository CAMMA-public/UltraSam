from typing import Dict, List, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear
from mmcv.cnn.bricks.transformer import FFN
from mmengine.model import BaseModule, Sequential
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.structures.mask import mask2bbox
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.utils import (ConfigType, InstanceList, OptInstanceList,
                         OptMultiConfig, reduce_mean)
from mmdet.models.utils import multi_apply
from mmdet.models.dense_heads import DeformableDETRHead
from mmdet.models.layers.transformer import MLP
from endosam.models.task_modules.prior_generators.prompt_encoder import EmbeddingIndex

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

@MODELS.register_module()
class SAMHead(BaseModule):
    """
    Head of SAM. Segment Anything.

    More details can be found in the `paper
    <https://arxiv.org/pdf/2304.02643>`_ .

    Losses and training. We supervise mask prediction with
    the linear combination of focal loss [65] and dice loss [73]
    used in [14]. We train for the promptable segmentation task
    using a mixture of geometric prompts (for text prompts see
    §7.5). Following [92, 37], we simulate an interactive setup
    by randomly sampling prompts in 11 rounds per mask, al-
    lowing SAM to integrate seamlessly into our data engine.
    to rank masks, the model pre-
    dicts a confidence score (i.e., estimated IoU) for each mask.

    During training, we compute the loss (described shortly) between
    the ground truth and each of the predicted masks, but only
    backpropagate from the lowest loss
    For use in applications, we’d like to rank predicted masks,
    so we add a small head (operating on an additional output
    token) that estimates the IoU between each predicted mask
    and the object it covers.

    Args:
        transformer_dim (int): the channel dimension of the transformer
            transformer (nn.Module): the transformer used to predict masks
        num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
        activation (nn.Module): the type of activation to use when
            upscaling masks
        iou_head_depth (int): the depth of the MLP used to predict
            mask quality
        iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        loss_score (:obj:`ConfigDict` or dict): Config of the score
            loss. Defaults to `CrossEntropyLoss`.
        loss_iou (:obj:`ConfigDict` or dict): Config of the regression iou
            loss. Defaults to `GIoULoss`.
        train_cfg (:obj:`ConfigDict` or dict): Training config of transformer
            head.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(
            self,
            *,
            transformer_dim: int,
            num_multimask_outputs: int = 3,
            activation: Type[nn.Module] = nn.GELU,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 256,
            loss_iou_score: ConfigType = dict(
                type='MSELoss',
                loss_weight=1.0,
            ),
            loss_mask_focal: ConfigType = dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=20.0),
            loss_mask_dice: ConfigType = dict(
                type='DiceLoss',
                use_sigmoid=True,
                activate=True,
                reduction='mean',
                eps=1.0,
                loss_weight=1.0),
            train_cfg: ConfigType = dict(
                assigner=dict(type='SAMassigner',)
                ),
            init_cfg: OptMultiConfig = None,
            **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        self.transformer_dim = transformer_dim
        self.activation = activation
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim

        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1

        self.train_cfg = train_cfg
        self.loss_iou_score = MODELS.build(loss_iou_score)
        self.loss_mask_focal = MODELS.build(loss_mask_focal)
        self.loss_mask_dice = MODELS.build(loss_mask_dice)

        if self.loss_mask_focal.use_sigmoid:
            self.score_out_channels = 1
        else:
            self.score_out_channels = 2

        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize layers of the transformer head."""
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(self.transformer_dim, self.transformer_dim // 4, kernel_size=2, stride=2),  # noqa
            LayerNorm2d(self.transformer_dim // 4),
            self.activation(),
            nn.ConvTranspose2d(self.transformer_dim // 4, self.transformer_dim // 8, kernel_size=2, stride=2),  # noqa
            self.activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(self.transformer_dim, self.transformer_dim, self.transformer_dim // 8, 3) # noqa
                for _ in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            self.transformer_dim, self.iou_head_hidden_dim, self.num_mask_tokens, self.iou_head_depth  # noqa
        )

    def forward(
        self,
        shape: Tuple,
        point_embedding: Tensor,
        image_embedding: Tensor,
        padded_points: Tensor,
        padded_labels: Tensor,
        prompt_padding_masks: Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts masks. See 'forward' for more details.
            point_embedding (torch.Tensor): the processed point_embedding
            image_embedding (torch.Tensor): the processed image_embedding
        """
        b, c, h, w = shape
        # remove padded querries
        active_prompts = ~(prompt_padding_masks.bool())  # per instance & prompt padding
        active_inputs = ~torch.all(~active_prompts, 1)  # per instance padding

        act_prompt_padding_masks = ~(prompt_padding_masks[active_inputs].bool())
        act_point_embedding = point_embedding[active_inputs]
        act_image_embedding = image_embedding[active_inputs]
        act_padded_points = padded_points[active_inputs]
        act_padded_labels = padded_labels[active_inputs]
        # is_mask_token = act_padded_labels == EmbeddingIndex.MASK_OUT.value
        # is_iou_token = act_padded_labels == EmbeddingIndex.IOU_OUT.value

        act_b = active_inputs.sum()

        # 1. extract output token
        # mask_token = act_point_embedding[is_mask_token].view(act_b, self.num_mask_tokens, -1)
        # iou_token = act_point_embedding[is_iou_token]
        mask_tokens_out = act_point_embedding[:, -(self.num_mask_tokens+1):-1]
        iou_token_out = act_point_embedding[:, -1]  # iou token is last

        # 2. Upscale mask embeddings and predict masks using the mask tokens
        act_image_embedding = act_image_embedding.transpose(1, 2).view(act_b, c, h, w)
        upscaled_embedding = self.output_upscaling(act_image_embedding)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))  # noqa
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape

        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)  # noqa

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)

        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred


    # def forward(
    #     self,
    #     shape: Tuple,
    #     point_embedding: Tensor,
    #     image_embedding: Tensor,
    #     padded_points: Tensor,
    #     padded_labels: Tensor,
    #     prompt_padding_masks: Tensor,
    #     multimask_output: bool,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Predicts masks. See 'forward' for more details.
    #         point_embedding (torch.Tensor): the processed point_embedding
    #         image_embedding (torch.Tensor): the processed image_embedding
    #     """
    #     # b, c, h, w = shape
    #     ori_b, n_inst, c, h, w = shape
    #     # remove padded querries
    #     active_prompts = ~(prompt_padding_masks.bool())  # per instance & prompt padding
    #     active_inputs = ~torch.all(~active_prompts, 1)  # per instance padding

    #     act_prompt_padding_masks = ~(prompt_padding_masks[active_inputs].bool())
    #     act_point_embedding = point_embedding[active_inputs]
    #     act_image_embedding = image_embedding[active_inputs]
    #     act_padded_points = padded_points[active_inputs]
    #     act_padded_labels = padded_labels[active_inputs]
    #     # is_mask_token = act_padded_labels == EmbeddingIndex.MASK_OUT.value
    #     # is_iou_token = act_padded_labels == EmbeddingIndex.IOU_OUT.value

    #     act_b = active_inputs.sum()

    #     # 1. extract output token
    #     # mask_token = act_point_embedding[is_mask_token].view(act_b, self.num_mask_tokens, -1)
    #     # iou_token = act_point_embedding[is_iou_token]
    #     mask_tokens_out = act_point_embedding[:, -(self.num_mask_tokens+1):-1]
    #     iou_token_out = act_point_embedding[:, -1]  # iou token is last

    #     # 2. Upscale mask embeddings and predict masks using the mask tokens
    #     act_image_embedding = act_image_embedding.transpose(1, 2).view(act_b, c, h, w)
    #     upscaled_embedding = self.output_upscaling(act_image_embedding)

    #     hyper_in_list: List[torch.Tensor] = []
    #     for i in range(self.num_mask_tokens):
    #         hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))  # noqa
    #     hyper_in = torch.stack(hyper_in_list, dim=1)
    #     b, c, h, w = upscaled_embedding.shape

    #     masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)  # noqa

    #     # Generate mask quality predictions
    #     iou_pred = self.iou_prediction_head(iou_token_out)

    #     # Select the correct mask or masks for output
    #     if multimask_output:
    #         mask_slice = slice(1, None)
    #     else:
    #         mask_slice = slice(0, 1)

    #     masks = masks[:, mask_slice, :, :]
    #     iou_pred = iou_pred[:, mask_slice]

    #     return masks, iou_pred

    def predict(
        self,
        shape: Tuple,
        point_embedding: Tensor,
        image_embedding: Tensor,
        padded_points: Tensor,
        padded_labels: Tensor,
        prompt_padding_masks: Tensor,
        batch_data_samples: SampleList,
        rescale: bool = True,
        multimask_output: bool = True,
    ):
        mask_logits, iou_preds = self(shape, point_embedding, image_embedding,
                    padded_points, padded_labels, prompt_padding_masks, multimask_output)

        # get masks from logits
        masks = mask_logits > 0

        if multimask_output:
            # select best mask based on iou_preds
            masks = masks.gather(1, iou_preds.max(1).indices.view(-1, 1, 1, 1).expand_as(masks)[:, 0:1])
            mask_logits = mask_logits.gather(1, iou_preds.max(1).indices.view(-1, 1, 1, 1).expand_as(mask_logits)[:, 0:1])
            scores = iou_preds.max(1).values
        else:
            scores = iou_preds.squeeze(1)

        results = []

        # split masks, scores, and mask_logits by img
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        instances_per_img = [len(b.gt_instances) for b in batch_data_samples]
        masks = list(masks.split(instances_per_img))
        scores = scores.split(instances_per_img)
        mask_logits = mask_logits.split(instances_per_img)

        for ind, b in enumerate(batch_img_metas):
            if rescale:
                pad_shape = b['pad_shape']
                img_shape = b['img_shape']
                ori_shape = b['ori_shape']

                # first resize mask to pad shape
                masks[ind] = F.interpolate(masks[ind].float(), pad_shape, mode="bilinear",
                        align_corners=False)

                # now crop to img shape
                masks[ind] = masks[ind][..., :img_shape[0], :img_shape[1]]

                # finally resize to ori_shape
                masks[ind] = F.interpolate(masks[ind], ori_shape, mode="bilinear",
                        align_corners=False)

            # cast masks and extract bboxes
            masks[ind] = masks[ind].squeeze(1).bool()
            bboxes = mask2bbox(masks[ind])

            result = InstanceData()
            result.bboxes = bboxes
            result.labels = batch_data_samples[ind].gt_instances.labels
            result.scores = scores[ind]
            result.masks = masks[ind]
            result.mask_logits = mask_logits[ind]
            # results.append(InstanceData(masks=masks[ind], bboxes=bboxes, scores=scores[ind],
            #     # point_pred=b.gt_instances[0].points,
            #     labels=b.gt_instances.labels, mask_logits=mask_logits[ind]))
            results.append(result)

        return results

    def loss(
        self,
        shape: Tuple,
        point_embedding: Tensor,
        image_embedding: Tensor,
        padded_points: Tensor,
        padded_labels: Tensor,
        prompt_padding_masks: Tensor,
        batch_data_samples: SampleList,
        multimask_output: bool = True,
    ) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            hidden_states (Tensor): Feature from the transformer decoder, has
                shape (num_decoder_layers, bs, num_queries, cls_out_channels)
                or (num_decoder_layers, num_queries, bs, cls_out_channels).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(shape, point_embedding, image_embedding,
                    padded_points, padded_labels, prompt_padding_masks,
                    multimask_output)

        mask_logits, iou_preds = outs

        # masks [b, 1 or 3, 256, 256]
        # iou [b, 1 or 3]
        loss_inputs = outs + (batch_gt_instances, batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs)

        return losses, mask_logits

    def loss_by_feat(
        self,
        mask_preds: Tensor,
        iou_pred: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        The IoU prediction head is trained with mean-square-error loss
        between the IoU prediction and the predicted mask’s IoU with the ground truth mask

        Returns:
            Tuple[Tensor]: A tuple including `loss_score`, `loss_iou`.
        """

        # get targets
        selected_pred_masks, selected_pred_ious, selected_gt_ious, resized_gt_masks = \
                self._get_targets_single(mask_preds, iou_pred, batch_gt_instances)

        # dict for losses
        losses = {}

        # compute dice loss
        loss_dice = self.loss_mask_dice(selected_pred_masks, resized_gt_masks)
        losses.update({'loss_dice': loss_dice})

        # compute focal loss
        loss_focal = self.loss_mask_focal(selected_pred_masks, resized_gt_masks)
        losses.update({'loss_focal': loss_focal})

        # compute iou loss
        loss_iou = self.loss_iou_score(selected_pred_ious, selected_gt_ious)
        losses.update({'loss_iou': loss_iou})

        return losses

    def _get_targets_single(
        self,
        mask_preds: Tensor,
        iou_pred: Tensor,
        gt_instances: InstanceData) -> tuple:
        """rescale mask, compute IOU as target for iou token,
        only best mask is returned
        """

        # get some dims
        B, num_masks, H, W = mask_preds.shape

        # get gt masks
        gt_masks = [b.masks.to_tensor(device=mask_preds.device, dtype=mask_preds.dtype) for b in gt_instances]

        # pad gt masks
        # NOTE could be done in preprocessor, but need to be done only on training data
        target_size = max(gt_instances[0].masks.width, gt_instances[0].masks.height)
        padded_gt_masks = torch.cat([F.pad(m, (0, target_size - m.shape[2], 0,
            target_size - m.shape[1])) for m in gt_masks])

        # resize gt masks
        resized_gt_masks = F.interpolate(padded_gt_masks.unsqueeze(0),
                size=mask_preds.shape[-2:], mode='bilinear').squeeze(0).round()

        # compute and select ious
        ious = self.compute_mask_iou(mask_preds, resized_gt_masks)
        selected_gt_ious, selected_mask_inds = ious.max(-1)
        selected_pred_ious = torch.gather(iou_pred, 1, selected_mask_inds.unsqueeze(-1).long()).squeeze(-1)

        # select pred masks based on highest iou for each instance
        selected_pred_masks = torch.gather(mask_preds, 1, selected_mask_inds.long().view(
            B, 1, 1, 1).repeat(1, 1, H, W)).squeeze(1)

        return selected_pred_masks, selected_pred_ious, selected_gt_ious, resized_gt_masks

    def compute_mask_iou(
        self,
        pred_masks: Tensor,
        gt_masks: Tensor) -> Tensor:
        """
        Compute the IoU between each predicted mask and the ground truth mask.

        Args:
            pred_masks (torch.Tensor): Predicted masks of shape (B, 3, H, W), with values between 0 and 1.
            gt_masks (torch.Tensor): Ground truth masks of shape (B, H, W), binary masks with values 0 or 1.

        Returns:
            torch.Tensor: IoU values of shape (B, 3) for each predicted mask compared to the ground truth mask.
        """
        # B: batch size, num_masks: number of predicted masks, H: height, W: width
        B, num_masks, H, W = pred_masks.shape

        # Expand the ground truth mask to match the shape of predicted masks
        gt_masks_expanded = gt_masks.unsqueeze(1).expand(-1, num_masks, -1, -1)  # Shape: (B, num_masks, H, W)

        # Convert the predicted masks to binary masks (0 or 1) using a threshold of 0.5
        pred_binary = (pred_masks.sigmoid() > 0.5).float()  # Shape: (B, num_masks, H, W)

        # Compute the intersection: element-wise AND between predicted and ground truth masks
        intersection = (pred_binary * gt_masks_expanded).sum(dim=(2, 3))  # Sum over H and W, result shape: (B, num_masks)

        # Compute the union: element-wise OR, which is equivalent to the sum of areas minus the intersection
        union = pred_binary.sum(dim=(2, 3)) + gt_masks_expanded.sum(dim=(2, 3)) - intersection  # Shape: (B, num_masks)

        # Avoid division by zero by setting union to a small value if it's zero
        union = torch.clamp(union, min=1e-6)

        # Compute IoU
        iou = intersection / union  # Shape: (B, num_masks)

        return iou
