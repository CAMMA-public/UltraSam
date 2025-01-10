from mmpretrain.models.backbones import ViTSAM
from mmdet.registry import MODELS
from typing import Optional, Sequence, Tuple
from mmpretrain.models.utils import LayerNorm2d, build_norm_layer, resize_pos_embed
import torch


@MODELS.register_module()
class ViTSAMcls(ViTSAM):
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)
        x = x.view(B, patch_resolution[0], patch_resolution[1],
                   self.embed_dims)

        if self.use_abs_pos:
            # 'resize_pos_embed' only supports 'pos_embed' with ndim==3, but
            # in ViTSAM, the 'pos_embed' has 4 dimensions (1, H, W, C), so it
            # is flattened. Besides, ViTSAM doesn't have any extra token.
            resized_pos_embed = resize_pos_embed(
                self.pos_embed.flatten(1, 2),
                self.patch_resolution,
                patch_resolution,
                mode=self.interpolate_mode,
                num_extra_tokens=0)
            x = x + resized_pos_embed.view(1, *patch_resolution,
                                           self.embed_dims)
            x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i in self.out_indices:
                # (B, H, W, C) -> (B, C, H, W)
                x_reshape = x.permute(0, 3, 1, 2)

                if self.out_channels > 0:
                    x_reshape = self.channel_reduction(x_reshape)
                outs.append(self._format_output(x_reshape))

        return tuple(outs), x.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1).mean(dim=1)
