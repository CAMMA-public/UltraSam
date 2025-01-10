_base_ = './sam_classification.py'

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmpretrain.LoRAModel',
        module=dict(
            type='mmpretrain.ViTSAM',
            arch='base',
            img_size=1024,
            patch_size=16,
            out_channels=0,
            use_abs_pos=True,
            use_rel_pos=True,
            window_size=14,
            out_type="avg_featmap",
            init_cfg=dict(
                type='Pretrained',
                prefix="backbone.",
                checkpoint="weights/mapped_sam_vit_b_01ec64.pth"
            ),
        ),
        alpha=16,
        rank=16,
        drop_rate=0.1,
        targets=[dict(type='qkv')]
    ),
)
