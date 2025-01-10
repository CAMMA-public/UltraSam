from endosam.models.dense_heads.sam_mask_decoder import SAMHead, LayerNorm2d
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


@MODELS.register_module()
class SamClassificationHead(SAMHead):
    """
    SAM Head with classification head.
    Allow to finetum SAM and predict instance class.
    prompt-based classification
    """
    def __init__(
        self,
        *,
        num_classes: int = 2,
        loss_cls: ConfigType = dict(
                type='mmpretrain.FocalLoss',
                loss_weight=1.0,
        ),
        **kwargs,
    ) -> None:
        self.num_classes = num_classes
        super().__init__(**kwargs)
        self.loss_cls = MODELS.build(loss_cls)

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
        # self.cls_prediction_head = Linear(self.transformer_dim, self.num_classes)
        self.cls_prediction_head = MLP(
            self.transformer_dim*2+768,
            # self.transformer_dim+768,
            self.iou_head_hidden_dim,
            # 512,
            self.num_classes, 2
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
        raw_img_feats: Tensor
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

        act_point_embedding = point_embedding[active_inputs]
        act_image_embedding = image_embedding[active_inputs]

        act_b = active_inputs.sum()

        # 1. extract output token
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
        # TODO maybe try to add a classification token?
        # TODO predict from image embedding
        _, img_embed_c, img_embed_h, img_embed_w = act_image_embedding.shape

        act_image_embedding = act_image_embedding.view(b, img_embed_c, img_embed_h*img_embed_w)
        act_image_embedding = act_image_embedding.transpose(1, 2)
        mean_img_embedding = act_image_embedding.mean(dim=1)
        
        # print(raw_img_feats.shape)
        # feat_map_embed.view(b, -1, img_embed_c, img_embed_h, img_embed_w).shape
        # torch.Size([8, 20, 256, 64, 64])
        # breakpoint()

        concatenated_tensor = torch.cat((mask_tokens_out[:, 0, :], mean_img_embedding, raw_img_feats), dim=1)
        # TODO predict from the decoder feature only, not using the prompt
        # concatenated_tensor = torch.cat((mask_tokens_out[:, 0, :], raw_img_feats), dim=1)
        # concatenated_tensor = mean_img_embedding

        cls_score = self.cls_prediction_head(concatenated_tensor)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)

        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        # cls_score = cls_score[:, 0, :]

        return masks, iou_pred, cls_score

    def predict(
        self,
        shape: Tuple,
        point_embedding: Tensor,
        image_embedding: Tensor,
        padded_points: Tensor,
        padded_labels: Tensor,
        prompt_padding_masks: Tensor,
        batch_data_samples: SampleList,
        raw_img_feats: Tensor,
        rescale: bool = True,
        multimask_output: bool = True,
    ):
        mask_logits, iou_preds, cls_score = self(shape, point_embedding, image_embedding,
                    padded_points, padded_labels, prompt_padding_masks, multimask_output, raw_img_feats)

        # get masks from logits
        masks = mask_logits > 0

        if multimask_output:
            # select best mask based on iou_preds
            masks = masks.gather(1, iou_preds.max(1).indices.view(-1, 1, 1, 1).expand_as(masks)[:, 0:1])
            mask_logits = mask_logits.gather(1, iou_preds.max(1).indices.view(-1, 1, 1, 1).expand_as(mask_logits)[:, 0:1])
            scores = iou_preds.max(1).values
        else:
            scores = iou_preds.squeeze(1)

        cls_scores = F.sigmoid(cls_score)  # .max(-1)
        # cls_scores = F.softmax(cls_score, dim=-1)  # .max(-1)
        # cls_scores, cls_labels = F.softmax(cls_score, dim=-1).max(-1)
        # breakpoint()
        bboxes = mask2bbox(masks)

        if rescale:
            pad_shape = batch_data_samples[0].pad_shape
            img_shape = batch_data_samples[0].img_shape
            ori_shape = batch_data_samples[0].ori_shape

            # first resize mask to pad shape
            masks = F.interpolate(masks.float(), pad_shape, mode="bilinear", align_corners=False)

            # now crop to img shape
            masks = masks[..., :img_shape[0], :img_shape[1]]

            # finally resize to ori_shape
            masks = F.interpolate(masks, ori_shape, mode="bilinear", align_corners=False)

        masks = masks.squeeze(1).bool()

        # get boxes from masks
        bboxes = mask2bbox(masks)
        # print(cls_labels, batch_data_samples[0].gt_instances.labels)

        # NOTE only works with batch size 1
        results = InstanceData(
            masks=masks, bboxes=bboxes, scores=scores,
            pred_score=cls_scores,
            labels=batch_data_samples[0].gt_instances.labels, mask_logits=mask_logits)
        # print(cls_scores, batch_data_samples[0].gt_instances.labels)

        return [results]

    def loss(
        self,
        shape: Tuple,
        point_embedding: Tensor,
        image_embedding: Tensor,
        padded_points: Tensor,
        padded_labels: Tensor,
        prompt_padding_masks: Tensor,
        batch_data_samples: SampleList,
        raw_img_feats = None,
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
                    multimask_output, raw_img_feats)

        mask_logits, iou_preds, cls_prd = outs

        # masks [b, 1 or 3, 256, 256]
        # iou [b, 1 or 3]
        loss_inputs = outs + (batch_gt_instances, batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs)

        return losses, mask_logits

    def loss_by_feat(
        self,
        mask_preds: Tensor,
        iou_pred: Tensor,
        cls_prd: Tensor,
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
        selected_pred_masks, selected_pred_ious, selected_gt_ious, resized_gt_masks, selected_gt_cls = \
                self._get_targets_single(mask_preds, iou_pred, cls_prd, batch_gt_instances)

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

        # compute cls loss
        # print(F.sigmoid(cls_prd), selected_gt_cls)
        loss_cls = self.loss_cls(cls_prd, selected_gt_cls)
        losses.update({'loss_cls': loss_cls})

        return losses

    def _get_targets_single(
        self,
        mask_preds: Tensor,
        iou_pred: Tensor,
        cls_prd: Tensor,
        gt_instances: InstanceData) -> tuple:
        """rescale mask, compute IOU as target for iou token,
        only best mask is returned
        """

        # get some dims
        B, num_masks, H, W = mask_preds.shape

        # get gt masks
        gt_masks = [b.masks.to_tensor(device=mask_preds.device, dtype=mask_preds.dtype) for b in gt_instances]

        # pad gt masks
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

        selected_gt_cls = torch.cat([b.labels for b in gt_instances])

        return selected_pred_masks, selected_pred_ious, selected_gt_ious, resized_gt_masks, selected_gt_cls