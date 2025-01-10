import torch
import numpy as np
import sys

ckpt_names = {
    'ViT-B': 'sam_vit_b_01ec64',
    'ViT-L': 'sam_vit_l_0b3195',
    'ViT-H': 'sam_vit_h_4b8939',
    'MedSAM': 'medsam_vit_b',
    'SAM-Med2D': 'sam-med2d_b',
    'FT-SAM': 'ft-sam_b',
}

if len(sys.argv) != 2:
    raise ValueError("Must specify SAM variant as one of [ViT-B/ViT-L/ViT-H/MedSAM/SAM-Med2D/FT-SAM]")
variant = sys.argv[1]

# load both weights
mmdet_sam_wts = torch.load('ref/mmdet_{}_state_dict.pth'.format(ckpt_names[variant]))
mapped_bb_wts = torch.load('mmdet_{}_backbone.pth'.format(ckpt_names[variant])) # load sam bb wts
orig_sam_wts = torch.load('{}.pth'.format(ckpt_names[variant]))
if 'model' in orig_sam_wts.keys():
    orig_sam_wts = orig_sam_wts['model']
mapped_sam_wts = {'state_dict': {}, 'meta': {}}

# add mapped bb wts to mapped state dict
mapped_sam_wts['state_dict'].update(mapped_bb_wts['state_dict'])

# decoder and mask head
sam_decoder_mask_head_wts = {k: v for k, v in orig_sam_wts.items() if 'mask_decoder.' in k}

# separate decoder and mask head weights from sam and map decoder weights
sam_decoder_wts = {k.replace('mask_decoder.transformer.', 'decoder.'): v \
        for k, v in sam_decoder_mask_head_wts.items() if 'mask_decoder.transformer' in k}
sam_mask_head_wts = {k.replace('mask_decoder.', 'bbox_head.'): v \
        for k, v in sam_decoder_mask_head_wts.items() if 'mask_decoder.transformer' not in k}

# map decoder weights
sam_decoder_wts = {k.replace('mlp.lin1', 'mlp.layers.0.0').replace('mlp.lin2', 'mlp.layers.1').replace(
        'norm_final_attn', 'post_norm'): v for k, v in sam_decoder_wts.items()}
sam_decoder_wts = {k.replace('cross_attn_image_to_token', 'cross_attn_image_to_token.attn').replace(
    'cross_attn_token_to_image', 'cross_attn_token_to_image.attn').replace(
        'final_attn_token_to_image', 'final_attn_token_to_image.attn').replace(
            'self_attn', 'self_attn.attn').replace(
                'proj.', 'proj_').replace('out_proj_', 'out_proj.'): v \
                        for k, v in sam_decoder_wts.items()}
# need to combine the bias terms into a single 'in_proj_bias' tensor
attn_layer_names = ['layers.0.self_attn', 'layers.0.cross_attn_token_to_image',
        'layers.0.cross_attn_image_to_token', 'layers.1.self_attn',
        'layers.1.cross_attn_token_to_image', 'layers.1.cross_attn_image_to_token',
        'final_attn_token_to_image']
for l in attn_layer_names:
    sam_decoder_wts['decoder.{}.attn.in_proj_bias'.format(l)] = torch.cat([
        sam_decoder_wts.pop('decoder.{}.attn.q_proj_bias'.format(l)),
        sam_decoder_wts.pop('decoder.{}.attn.k_proj_bias'.format(l)),
        sam_decoder_wts.pop('decoder.{}.attn.v_proj_bias'.format(l)),
    ])

mapped_sam_wts['state_dict'].update(sam_decoder_wts)

# map prompt encoder weights
sam_prompt_encoder_wts = {k: v for k, v in orig_sam_wts.items() if 'prompt_encoder.' in k}

# combine all the nn.embeddings into 1 stacked embedding NOTE ignoring not_a_point for now
#sam_prompt_encoder_wts.pop('prompt_encoder.no_mask_embed.weight'),
combined_label_embedding = torch.cat([
    sam_prompt_encoder_wts.pop('prompt_encoder.no_mask_embed.weight'),
    sam_prompt_encoder_wts.pop('prompt_encoder.point_embeddings.0.weight'),
    sam_prompt_encoder_wts.pop('prompt_encoder.point_embeddings.1.weight'),
    sam_prompt_encoder_wts.pop('prompt_encoder.point_embeddings.2.weight'),
    sam_prompt_encoder_wts.pop('prompt_encoder.point_embeddings.3.weight'),
    sam_prompt_encoder_wts.pop('prompt_encoder.not_a_point_embed.weight'),
    sam_mask_head_wts.pop('bbox_head.mask_tokens.weight'), # 4 mask output tokens
    sam_mask_head_wts.pop('bbox_head.iou_token.weight')
    ])
sam_prompt_encoder_wts['prompt_encoder.label_encoder.label_embedding.weight'] = combined_label_embedding

# rename pe layer to be in main detector
mapped_sam_wts['state_dict'].update(sam_prompt_encoder_wts)

# repeat for mask_head_wts
mapped_sam_wts['state_dict'].update(sam_mask_head_wts)

missing_keys = [k for k in mmdet_sam_wts.keys() if k not in mapped_sam_wts['state_dict']]
unmatched_keys = [k for k in mapped_sam_wts['state_dict'].keys() if k not in mmdet_sam_wts.keys()]

print("MISSING {} KEYS IN SOURCE STATE DICT:".format(len(missing_keys)), missing_keys)
print("{} UNMATCHED KEYS FROM SOURCE STATE DICT:".format(len(unmatched_keys)), unmatched_keys)

mapped_ckpt_name = 'mapped_{}.pth'.format(ckpt_names[variant])
print("SAVING MAPPED CHECKPOINT TO", mapped_ckpt_name)
torch.save(mapped_sam_wts, mapped_ckpt_name)
