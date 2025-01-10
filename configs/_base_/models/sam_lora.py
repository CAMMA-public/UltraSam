_base_ = './sam_mask_refinement.py'

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmpretrain.LoRAModel',
        module=dict(
            type='mmpretrain.ViTSAM',
            arch='base',
            img_size=1024,
            patch_size=16,
            out_channels=256,
            use_abs_pos=True,
            use_rel_pos=True,
            window_size=14,
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

custom_hooks = [dict(type="MonkeyPatchHook"), dict(type="FreezeHook", freeze_all_but_lora=True)]

# model = dict(
#     _delete_=True,
#     type='mmpretrain.LoRAModel',
#     module=dict(
#         type='SAM',
#         data_preprocessor=_base_.data_preprocessor,
#         prompt_encoder=dict(
#             type='SAMPaddingGenerator',
#             label_encoder=dict(
#                 type='LabelEmbedEncoder',
#                 embed_dims=256,
#             ),
#         ),
#         bbox_head=dict(
#             type='SAMHead',
#         ),
#         backbone=dict(
#             type='mmpretrain.ViTSAM',
#             arch='base',
#             img_size=1024,
#             patch_size=16,
#             out_channels=256,
#             use_abs_pos=True,
#             use_rel_pos=True,
#             window_size=14,
#         ),
#         decoder=dict(  # SAMTransformerDecoder
#             num_layers=2,
#             layer_cfg=dict(  # SAMTransformerLayer
#                 embedding_dim=256,
#                 num_heads=8,
#                 ffn_cfg=dict(
#                     embed_dims=256,
#                     feedforward_channels=2048,
#                     ffn_drop=0.1
#                 ),
#             ),
#         ),
#         train_cfg=dict(
#             assigner=dict(
#                 type='SAMassigner',
#             )
#         ),
#     ),
#     alpha=16,
#     rank=16,
#     drop_rate=0.1,
#     targets=[dict(type='qkv')]
# )
