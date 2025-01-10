import torch
import sys

def check_mismatches(sd, target_sd):
    missing_keys = [k for k in target_sd.keys() if k not in sd.keys()]
    extra_keys = [k for k in sd.keys() if k not in target_sd.keys()]

    print("MISSING {} KEYS:".format(len(missing_keys)), missing_keys)
    print("{} EXTRA KEYS:".format(len(extra_keys)), extra_keys)

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

sam_wts = torch.load('{}.pth'.format(ckpt_names[variant]))
if 'model' in sam_wts.keys():
    sam_wts = sam_wts['model']

sam_bb_wts = {k.replace('image_encoder.', ''): v for k, v in sam_wts.items() if 'image_encoder' in k}

mmdet_state_dict = torch.load('ref/mmdet_{}_state_dict.pth'.format(ckpt_names[variant]))
mmdet_bb_state_dict = {k.replace('backbone.', ''): v for k, v in mmdet_state_dict.items() if 'backbone' in k}

# blocks -> layers
sam_bb_wts = {k.replace('blocks.', 'layers.'): v for k, v in sam_bb_wts.items()}

# patch_embed.proj -> patch_embed.projection
sam_bb_wts = {k.replace('patch_embed.proj', 'patch_embed.projection'): v for k, v in sam_bb_wts.items()}

# ln -> norm
sam_bb_wts = {k.replace('norm', 'ln'): v for k, v in sam_bb_wts.items()}

# neck -> channel_reduction
sam_bb_wts = {k.replace('neck', 'channel_reduction'): v for k, v in sam_bb_wts.items()}

# mlp.lin1 -> ffn.layers.0.0
sam_bb_wts = {k.replace('mlp.lin1', 'ffn.layers.0.0'): v for k, v in sam_bb_wts.items()}

# mlp.lin2 -> ffn.layers.1
sam_bb_wts = {k.replace('mlp.lin2', 'ffn.layers.1'): v for k, v in sam_bb_wts.items()}

check_mismatches(sam_bb_wts, mmdet_bb_state_dict)

# add 'backbone' prefix to keys
sam_bb_wts = {'backbone.{}'.format(k): v for k, v in sam_bb_wts.items()}

sam_bb_wts_final = {'state_dict': sam_bb_wts, 'meta': {'mmpretrain_version': '1.0.0rc7', 'dataset_meta': {}}}
torch.save(sam_bb_wts_final, 'mmdet_{}_backbone.pth'.format(ckpt_names[variant]))
