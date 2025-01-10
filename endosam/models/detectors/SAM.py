import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.structures.bbox import scale_boxes
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig, InstanceList
from mmdet.models.detectors import DetectionTransformer
from mmengine.structures.pixel_data import PixelData
from ..utils.sam_layers import PositionEmbeddingRandom
from endosam.datasets.transforms.point_formatting import PackPointDetInputs
from endosam.models.utils.sam_layers import SAMTransformerDecoder
from endosam.datasets.transforms.custom_pipeline import PromptType
from endosam.models.utils.visualization import dump_masks, dump_fmap
from mmdet.structures.bbox import BaseBoxes
from mmengine.structures import InstanceData
from mmdet.models.utils import unpack_gt_instances


@MODELS.register_module()
class SAM(DetectionTransformer):
    """
        SAM predicts object masks from an image and input prompts.

        Arguments:
            backbone (ViTSAM): SAM backbone
            decoder (SAMTransformerDecoder): Transformer Decoder for SAM
            bbox_head (SAMHead): SAM Mask Prediction Head
            prompt_encoder (PromptEncoder): Encodes various types of input prompts.
    """

    def __init__(self,
            prompt_encoder: ConfigType,
            bbox_head: OptConfigType = None,
            use_mask_refinement: bool = False,
            num_mask_refinements: int = 1,
            **kwargs):
        bbox_head.update(transformer_dim=prompt_encoder['label_encoder']['embed_dims'])  # noqa
        super().__init__(bbox_head=bbox_head, **kwargs)
        prompt_encoder.update(use_mask_refinement=use_mask_refinement)
        self.prompt_encoder = MODELS.build(prompt_encoder)
        self.use_mask_refinement = use_mask_refinement
        self.num_mask_refinements = num_mask_refinements

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.decoder = SAMTransformerDecoder(**self.decoder)

    def encode_prompts(self, batch_data_samples: List[PackPointDetInputs], use_mask_prompt: bool) -> Dict:
        # get points from batch_data_samples
        padding_masks, attn_mask, padded_points, padded_labels, pts_embed, dense_embed = \
                self.prompt_encoder(batch_data_samples, encode_mask=use_mask_prompt)

        # pack into dict
        encoded_prompt_dict = dict(
                prompt_padding_masks=padding_masks,  # [bs, num_instance, num_points] 1 indicate is padding
                attn_mask=attn_mask,  # dont use
                padded_points=padded_points,  # [bs, num_instance, num_points, xy] points_coords
                padded_labels=padded_labels,  # [bs, num_instance, num_points] type of queries
                pts_embed=pts_embed,  # [bs, num_instance, num_points, embed_dim] embedding, including pos
                dense_embed=dense_embed,  # [bs, num_instance, embed_dim, w, h]
        )

        return encoded_prompt_dict

    def pre_transformer(
            self,
            img_feats: Tuple[Tensor],
            batch_data_samples: List[PackPointDetInputs],
            use_mask_prompt: bool) -> Tuple[Dict, Dict]:
        """
        Prepare inputs to transformer decoder by running prompt encoder and
        creating positional embeddings.
        """

        # get relevant img feats (not multiscale so just last feat map)
        feat = img_feats[-1]
        batch_size, feat_dim, _, _ = feat.shape

        # get img shape info
        batch_input_shape = batch_data_samples[0].batch_input_shape
        input_img_h, input_img_w = batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]

        # compute img padding mask
        padding_mask = feat.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = img_shape_list[img_id]
            padding_mask[img_id, :img_h, :img_w] = 0

        padding_mask = F.interpolate(
            padding_mask.unsqueeze(1),
            size=feat.shape[-2:]).to(torch.bool).squeeze(1)

        # run prompt encoder
        decoder_inputs_dict = self.encode_prompts(batch_data_samples, use_mask_prompt)

        # compute image pos embedding
        img_pos = self.prompt_encoder.get_dense_pe()

        # (Pdb) decoder_inputs_dict["dense_embed"].shape
        # torch.Size([2, 8, 256, 64, 64])
        # (Pdb) img_feats[-1].shape
        # torch.Size([2, 256, 64, 64])

        # store everything in decoder inputs dict
        decoder_inputs_dict.update(dict(img_feats=feat, img_pos=img_pos, padding_mask=padding_mask))

        return {}, decoder_inputs_dict

    def forward_encoder(self, **kwargs) -> Dict:
        return {}

    def pre_decoder(self, **kwargs) -> Tuple[Dict, Dict]:
        return {}, {}

    def forward_decoder(self, img_feats: Tensor, img_pos: Tensor, pts_embed: Tensor,
            padded_points: Tensor, padded_labels: Tensor, attn_mask: Tensor = None,
            prompt_padding_masks: Tensor = None, dense_embed: Tensor = None,
            padding_mask: Tensor = None, **kwargs) -> Dict:
        """Forward with Transformer decoder.

        Args:

        Returns:
            dict: The dictionary of decoder outputs, which includes
        """
        bs, num_instance, num_query, embed_dim = pts_embed.shape

        # (Pdb) img_pos.shape
        # torch.Size([64, 56, 64])
        # (Pdb) pts_embed.shape
        # torch.Size([2, 20, 14, 64])
        # (Pdb) img_feats.shape
        # torch.Size([2, 256, 56, 64])

        # TODO all this is already done, its simply the encoded prompts pts_embed
        # # Concatenate output tokens
        # output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        # output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        # tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in instance direction to be per-mask
        # 1. expand in instance dim
        # 2. add dense embed (encoded mask)
        # [bs, embed_dim, w, h] -> [bs, num_instance, embed_dim, w, h]
        # src = img_feats[:, None, :, :, :].expand(bs, num_instance, embed_dim, w, h)
        # src = img_feats[:, None, :, :, :].repeat(1, num_instance, 1, 1, 1)
        src = img_feats[:, None, :, :, :].repeat_interleave(num_instance, dim=1)
        src = src + dense_embed

        # [bs, num_inst, embed_dim, w, h] -> [bs*num_inst, embed_dim, w, h]
        src_shape = src.shape
        src = src.view(-1, *src.shape[2:])

        # [bs, num_inst x num_pts x embed_dim] -> [bs*num_inst x num_pts x embed_dim]
        pts_embed = pts_embed.view(-1, *pts_embed.shape[2:])

        # repeat img_pos for number of instances
        img_pos = img_pos.unsqueeze(0).repeat_interleave(bs * num_instance, dim=0)
        padding_mask = padding_mask.repeat_interleave(num_instance, dim=0).flatten(1)

        point_emb, img_emb = self.decoder(
            src,
            img_pos,
            pts_embed,
            padding_mask=None,  # padding_mask,  # I believe its None in SAM repo
            prompt_padding_mask=None,  # prompt_padding_masks
        )
        # (Pdb) point_emb.shape
        # torch.Size([24, 14, 256])
        # (Pdb) img_emb.shape
        # torch.Size([24, 4096, 256])

        # TODO reshape padded_labels to [bs*num_inst x num_pts] ?
        # padded input could be removed before applying the transformer
        padded_labels = padded_labels.view(bs*num_instance, num_query)
        padded_points = padded_points.view(bs*num_instance, num_query, -1)
        prompt_padding_masks = prompt_padding_masks.view(bs*num_instance, *prompt_padding_masks.shape[2:])

        # padded label
        head_inputs_dict = dict(
            # shape=src_shape,  # [bs, n_inst, C, h, w]
            shape=src.shape,  # [bs*n_inst, C, h, w]
            point_embedding=point_emb,  # [bs*n_inst, n_prompt, C]
            image_embedding=img_emb,  # [bs*n_inst, img_token, C]
            padded_points=padded_points,  # [bs*n_inst, n_prompt, 2]
            padded_labels=padded_labels,  # [bs*n_inst, n_prompt]
            prompt_padding_masks=prompt_padding_masks,  # [bs*n_inst, n_prompt]
        )

        return head_inputs_dict

    def forward_transformer(self,
                            img_feats: Tuple[Tensor],
                            batch_data_samples: OptSampleList = None,
                            use_mask_prompt: bool = False) -> Dict:

        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples, use_mask_prompt)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(**encoder_outputs_dict)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)

        return head_inputs_dict

    def predict(self,
            batch_inputs: Tensor,
            batch_data_samples: SampleList,
            rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with optional mask refinement"""
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = unpack_gt_instances(batch_data_samples) # noqa
        if batch_gt_instances[0].points.numel() == 0:
            results = InstanceData()
            results.bboxes = torch.tensor([])
            results.masks = torch.tensor([])
            results.scores = torch.tensor([])
            results.labels = torch.tensor([])
            results.mask_logits = torch.tensor([])

            batch_data_samples = self.add_pred_to_datasample(
                batch_data_samples, [results])
            return batch_data_samples
        # print(batch_data_samples)
        # print("==")
        # print(f"detector: {len(batch_data_samples)}")
        # try do retrieve attention k for cutler

        # # TODO move to an hook latter
        # feat_out = {}
        # def hook_fn_forward_qkv(module, input, output):
        #     feat_out["qkv"] = output

        # self.backbone._modules["layers"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)
        # # (Pdb) print(self.backbone._modules["layers"][-1]._modules["attn"]._modules["qkv"])
        # # Linear(in_features=768, out_features=2304, bias=True)


        img_feats = self.extract_feat(batch_inputs)
        # (Pdb) feat_out["qkv"].shape
        # torch.Size([1, 64, 64, 2304])
        # bs = 1
        # nb_token = 64*64
        # nb_head = 12
        # feat_h, feat_w = 1024 // 16, 1024 // 16

        # # windows att:
        # # (Pdb) feat_out["qkv"].shape
        # # torch.Size([25, 14, 14, 2304])
        # # bigger than gobal, because some padding

        # # global att:
        # # (Pdb) feat_out["qkv"].shape
        # # torch.Size([1, 64, 64, 2304])

        # qkv = (
        #     feat_out["qkv"]
        #     .reshape(bs, nb_token, 3, nb_head, -1)
        #     .permute(2, 0, 3, 1, 4)
        # )
        # q, k, v = qkv[0], qkv[1], qkv[2]
        # # (Pdb) k.shape
        # # torch.Size([1, 12, 4096, 64])



        # # breakpoint()
        # k = k.transpose(1, 2).reshape(bs, nb_token, -1)
        # q = q.transpose(1, 2).reshape(bs, nb_token, -1)
        # v = v.transpose(1, 2).reshape(bs, nb_token, -1)
        # # (Pdb) feats = k.transpose(1, 2)
        # # (Pdb) feats.shape
        # # torch.Size([1, 768, 4096])
        # k = k.transpose(1, 2).reshape(bs, 768, nb_token)

        # # TODO save feature directly?
        # # [1, 256, 64, 64]
        # output_token = img_feats[0]
        # output_token = output_token.reshape(1, 256, nb_token)

        # img_name = batch_data_samples[0].img_path.split("/")[-1].split(".")[1]
        # # torch.save(k, f"/home2020/home/icube/ameyer/CutLER/maskcut/ultrasam/{img_name}_feat.pt")
        # # torch.save(batch_inputs, f"/home2020/home/icube/ameyer/CutLER/maskcut/ultrasam/{img_name}_img.pt")
        # torch.save(output_token, f"/home2020/home/icube/ameyer/CutLER/maskcut/ultrasam/{img_name}_feat.pt")
        # torch.save(batch_inputs, f"/home2020/home/icube/ameyer/CutLER/maskcut/ultrasam/{img_name}_img.pt")

        # breakpoint()

        head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples,
            multimask_output=not self.use_mask_refinement,
        )

        if self.use_mask_refinement:
            for i in range(self.num_mask_refinements):
                # add predicted masks to batch_data_samples
                for r, b in zip(results_list, batch_data_samples):
                    b.gt_instances.mask_props = r['mask_logits']

                    # replace point prompts with box prompts derived from predicted mask
                    replace_boxes = b.gt_instances.prompt_types == PromptType.POINT.value
                    pred_box_prompts = scale_boxes(r['bboxes'], b.metainfo['scale_factor']).view(
                            r['bboxes'].shape[0], 2, 2)
                    b.gt_instances.boxes = torch.stack([pred_b if replace_boxes[ind] else \
                            gt_b for ind, (pred_b, gt_b) in enumerate(zip(pred_box_prompts, b.gt_instances.boxes))])
                    b.gt_instances.prompt_types[:] = PromptType.BOX.value

                # Cascaded Post-refinement-1
                head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples,
                        use_mask_prompt=True)
                results_list = self.bbox_head.predict(
                        **head_inputs_dict,
                        rescale=rescale,
                        batch_data_samples=batch_data_samples,
                        multimask_output=(i == self.num_mask_refinements-1))

        # convert mask to semantic segmentation mask and store for metric computation
        # for b, r in zip(batch_data_samples, results_list):
        #     # convert pred
        #     inst_masks = r['masks']
        #     scores = r['scores']
        #     labels = r['labels'] + 1
        #     fg = (inst_masks > 0.5).any(0).float()
        #     pixel_to_label = (inst_masks * scores.view(-1, 1, 1)).argmax(0)
        #     sem_mask = labels[pixel_to_label] * fg
        #     # b.pred_sem_seg = PixelData(sem_seg=sem_mask.unsqueeze(0))

        #     # convert gt
        #     inst_masks = b.gt_instances.masks.to_tensor(dtype=inst_masks.dtype,
        #             device=inst_masks.device)
        #     fg = (inst_masks == 1).any(0).float()
        #     pixel_to_label = (inst_masks * scores.view(-1, 1, 1)).argmax(0)
        #     sem_mask = labels[pixel_to_label] * fg
        #     b.gt_sem_seg = PixelData(sem_seg=sem_mask.unsqueeze(0))
        # breakpoint()
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        # print(batch_data_samples)
        # print("\n====\n")

        return batch_data_samples

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (bs, dim, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        losses, mask_logits = self.bbox_head.loss(**head_inputs_dict,
                batch_data_samples=batch_data_samples,
                multimask_output=not self.use_mask_refinement)

        if self.use_mask_refinement:
            for i in range(self.num_mask_refinements):
                # split mask_logits by img
                instances_per_img = [len(b.gt_instances) for b in batch_data_samples]
                mask_logits = mask_logits.split(instances_per_img)

                # add predicted masks to batch_data_samples
                for m, b in zip(mask_logits, batch_data_samples):
                    b.gt_instances.mask_props = m

                # Cascaded Post-refinement
                head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples,
                        use_mask_prompt=True)
                losses_ref_stage_i, mask_logits = self.bbox_head.loss(**head_inputs_dict,
                        batch_data_samples=batch_data_samples,
                        multimask_output=(i == self.num_mask_refinements-1))
                losses_ref_stage_i = {'{}.{}'.format(k, i): v for k, v in losses_ref_stage_i.items()}
                losses.update(losses_ref_stage_i)

        return losses

    # def add_pred_to_datasample(self, data_samples: SampleList,
    #                            results_list: InstanceList) -> SampleList:
    #     """Add predictions to `DetDataSample`.

    #     Args:
    #         data_samples (list[:obj:`DetDataSample`], optional): A batch of
    #             data samples that contain annotations and predictions.
    #         results_list (list[:obj:`InstanceData`]): Detection results of
    #             each image.

    #     Returns:
    #         list[:obj:`DetDataSample`]: Detection results of the
    #         input images. Each DetDataSample usually contain
    #         'pred_instances'. And the ``pred_instances`` usually
    #         contains following keys.

    #             - scores (Tensor): Classification scores, has a shape
    #                 (num_instance, )
    #             - labels (Tensor): Labels of bboxes, has a shape
    #                 (num_instances, ).
    #             - bboxes (Tensor): Has a shape (num_instances, 4),
    #                 the last dimension 4 arrange as (x1, y1, x2, y2).
    #     """
    #     for data_sample, pred_instances in zip(data_samples, results_list):
    #         data_sample.pred_instances = pred_instances
    #     self.samplelist_boxtype2tensor(data_samples)
    #     return data_samples

    # @staticmethod
    # def samplelist_boxtype2tensor(batch_data_samples: SampleList) -> SampleList:
    #     for data_samples in batch_data_samples:
    #         if 'gt_instances' in data_samples:
    #             bboxes = data_samples.gt_instances.get('bboxes', None)
    #             masks = data_samples.gt_instances.get('masks', None)
    #             if isinstance(bboxes, BaseBoxes):
    #                 data_samples.gt_instances.bboxes = bboxes.tensor
    #             dtype = data_samples.gt_instances.bboxes.dtype
    #             device = data_samples.gt_instances.bboxes.device
    #             data_samples.gt_instances.masks = masks.to_tensor(dtype=dtype, device=device)
    #         if 'pred_instances' in data_samples:
    #             bboxes = data_samples.pred_instances.get('bboxes', None)
    #             masks = data_samples.pred_instances.get('masks', None)
    #             if isinstance(bboxes, BaseBoxes):
    #                 data_samples.pred_instances.bboxes = bboxes.tensor
    #             data_samples.pred_instances.masks = masks.float()
    #             # dtype = data_samples.pred_instances.bboxes.dtype
    #             # device = data_samples.pred_instances.bboxes.device
    #             # data_samples.pred_instances.masks = masks.to_tensor(dtype=dtype, device=device)
    #         if 'ignored_instances' in data_samples:
    #             bboxes = data_samples.ignored_instances.get('bboxes', None)
    #             masks = data_samples.ignored_instances.get('masks', None)
    #             if isinstance(bboxes, BaseBoxes):
    #                 data_samples.ignored_instances.bboxes = bboxes.tensor
    #             # dtype = data_samples.ignored_instances.bboxes.dtype
    #             # device = data_samples.ignored_instances.bboxes.device
    #             # data_samples.ignored_instances.masks = masks.to_tensor(dtype=dtype, device=device)
