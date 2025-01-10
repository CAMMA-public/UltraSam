from mmpretrain.models.backbones.vit_sam import ViTSAM, TransformerEncoderLayer, window_partition, window_unpartition
from mmdet.registry import MODELS
from typing import Optional, Sequence, Tuple
import torch.nn as nn
import numpy as np
from mmengine.model import BaseModule, ModuleList
from mmpretrain.models.utils import  LayerNorm2d, build_norm_layer, resize_pos_embed, to_2tuple
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
import torch


class AdapterTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(
            self,
            adapter_mlp_ratio: float = 0.25,
            adapter_scale: float = 0.5,
            **kwargs):
        """
        Additional parameters for space and MLP adapters.
        see https://github.com/SuperMedIntel/Medical-SAM-Adapter/blob/4dcdc63c65c52e4acda35b96a50ebd8943e16238/models/ImageEncoder/vit/adapter_block.py#L12

        Args:
            adapter_mlp_ratio (float): Ratio to determine the hidden dimension
                size for the adapters relative to the embedding dimensions.
            kwargs: Additional arguments passed to the base class.
        """
        # Initialize the base class
        super().__init__(**kwargs)

        # Compute the adapter hidden dimension
        adapter_hidden_dim = int(adapter_mlp_ratio * self.embed_dims)
        self.adapter_scale = adapter_scale

        # Initialize the space adapter
        self.space_adapter = FFN(
            embed_dims=self.embed_dims,
            feedforward_channels=adapter_hidden_dim,
            act_cfg=dict(type='GELU'),
            add_identity=True,  # Enable residual connections
        )

        # Initialize the MLP adapter
        self.mlp_adapter = FFN(
            embed_dims=self.embed_dims,
            feedforward_channels=adapter_hidden_dim,
            act_cfg=dict(type='GELU'),
            add_identity=False,  # No residual connections
        )

    def forward(self, x: torch.Tensor):
        shortcut = x
        x = self.ln1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)

        # 1. space adapter
        x = self.space_adapter(x)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x_normed = self.ln2(x)

        # 2. residual adaptation
        adaptation = self.adapter_scale * self.mlp_adapter(x_normed)

        x = self.ffn(self.ln2(x), identity=x)
        x = x + adaptation
        return x


@MODELS.register_module()
class MED_SA(ViTSAM):
    # have to cc the init, no factory in parent class
    def __init__(self,
                 arch: str = 'base',
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 out_channels: int = 256,
                 out_indices: int = -1,
                 out_type: str = 'raw',
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 qkv_bias: bool = True,
                 use_abs_pos: bool = True,
                 use_rel_pos: bool = True,
                 window_size: int = 14,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 frozen_stages: int = 11,
                 interpolate_mode: str = 'bicubic',
                 patch_cfg: dict = dict(),
                 layer_cfgs: dict = dict(),
                 init_cfg: Optional[dict] = None):
        super(ViTSAM, self).__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        self.global_attn_indexes = self.arch_settings['global_attn_indexes']
        self.img_size = to_2tuple(img_size)

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size

        # Set out type
        if out_type not in self.OUT_TYPES:
            raise ValueError(f'Unsupported `out_type` {out_type}, please '
                             f'choose from {self.OUT_TYPES}')
        self.out_type = out_type

        self.use_abs_pos = use_abs_pos
        self.interpolate_mode = interpolate_mode
        if use_abs_pos:
            # Set position embedding
            self.pos_embed = nn.Parameter(
                torch.zeros(1, *self.patch_resolution, self.embed_dims))
            self.drop_after_pos = nn.Dropout(p=drop_rate)
            self._register_load_state_dict_pre_hook(self._prepare_pos_embed)

        if use_rel_pos:
            self._register_load_state_dict_pre_hook(
                self._prepare_relative_position)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.num_layers)

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                window_size=window_size
                if i not in self.global_attn_indexes else 0,
                input_size=self.patch_resolution,
                use_rel_pos=use_rel_pos,
                norm_cfg=norm_cfg)
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(AdapterTransformerEncoderLayer(**_layer_cfg))

        self.out_channels = out_channels
        if self.out_channels > 0:
            self.channel_reduction = nn.Sequential(
                nn.Conv2d(
                    self.embed_dims,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(out_channels, eps=1e-6),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm2d(out_channels, eps=1e-6),
            )

        # freeze stages only when self.frozen_stages > 0
        self.frozen_stages = frozen_stages
        if self.frozen_stages > 0:
            self._freeze_stages()

    def _freeze_stages(self):
        """modified to not freeze adapter layers
        """
        # freeze position embedding
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False
        # set dropout to eval model
        self.drop_after_pos.eval()
        # freeze patch embedding
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False

        # Freeze layers, exept adapter ones
        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            # m.eval()
            for name, param in m.named_parameters():
                # Skip freezing parameters with "adapter"
                if "adapter" not in name:
                    param.requires_grad = False

        # freeze channel_reduction module
        if self.frozen_stages == self.num_layers and self.out_channels > 0:
            m = self.channel_reduction
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
