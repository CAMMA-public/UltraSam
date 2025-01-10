# Adapted from https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/transformer.py
from typing import Union, Tuple, Type, Optional
import torch
from torch import Tensor, nn
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine import ConfigDict
from mmengine.model import BaseModule, ModuleList
from mmdet.utils import ConfigType, OptConfigType
import numpy as np
import math

class SAMTransformerDecoder(BaseModule):
    """
    A transformer decoder that attends to an input image using
    queries whose positional embedding is supplied.

    Args:
        num_layers (int): number of layers in the transformer
        layer_cfg (:obj:`ConfigDict` or dict): the config of each encoder
            layer. All the layers will share the same config. Contains an
            embedding_dim, num_heads, mlp_dim, and activation.
    """
    def __init__(
            self,
            num_layers: int,
            layer_cfg: ConfigType,
            attention_downsample_rate: int = 2,
            norm_cfg: OptConfigType = dict(type='LN'),
            init_cfg: OptConfigType = None,
            ) -> None:

        super().__init__(init_cfg=init_cfg)
        self.num_layers = num_layers
        self.norm_cfg = norm_cfg
        self.layer_cfg = layer_cfg
        self.attention_downsample_rate = attention_downsample_rate
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            SAMTransformerLayer(skip_first_layer_pe=(i == 0), **self.layer_cfg)
            for i in range(self.num_layers)
        ])
        self.embedding_dim = self.layers[0].embedding_dim

        self.final_attn_token_to_image = SAMAttention(
            self.embedding_dim,
            num_heads=self.layer_cfg['num_heads'],
            downsample_rate=self.attention_downsample_rate
        )
        self.post_norm = build_norm_layer(self.norm_cfg, self.embedding_dim)[1]

    def forward(
            self,
            image_embedding: Tensor,
            image_pos: Tensor,
            query_pos: Tensor,
            padding_mask: Tensor = None,
            prompt_attn_mask: Tensor = None,
            prompt_padding_mask: Tensor = None,
            ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pos (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          query_pos (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed query
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pos = image_pos.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = query_pos
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=query_pos,
                key_pe=image_pos,
                padding_mask=padding_mask,
                prompt_attn_mask=prompt_attn_mask,
                prompt_padding_mask=prompt_padding_mask,
            )

        # Apply the final attention layer from the points to the image
        q = queries + query_pos
        k = keys + image_pos

        attn_out = self.final_attn_token_to_image(
            q,
            key=k,
            value=keys,
            padding_mask=padding_mask)
        queries = queries + attn_out
        queries = self.post_norm(queries)

        return queries, keys


class SAMTransformerLayer(BaseModule):
    """
    A transformer block with four layers: (1) self-attention of sparse
    inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
    block on sparse inputs, and (4) cross attention of dense inputs to sparse
    inputs.

    Arguments:
        embedding_dim (int): the channel dimension of the embeddings
        num_heads (int): the number of heads in the attention layers
        mlp_dim (int): the hidden dimension of the mlp block
        activation (nn.Module): the activation of the mlp block
        skip_first_layer_pe (bool): skip the PE on the first layer
    """

    def __init__(
                self,
                num_heads: int = 8,
                embedding_dim: int = 256,
                attention_downsample_rate: int = 2,
                skip_first_layer_pe: bool = False,
                ffn_cfg: OptConfigType = dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.,
                    act_cfg=dict(type='ReLU', inplace=True),
                ),
                norm_cfg: OptConfigType = dict(type='LN'),
                init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)
        self.norm_cfg = norm_cfg
        self.embedding_dim = embedding_dim
        self.self_attn = SAMAttention(embedding_dim, num_heads=num_heads)
        self.norm1 = build_norm_layer(self.norm_cfg, embedding_dim)[1]

        self.cross_attn_token_to_image = SAMAttention(
            embedding_dim, num_heads=num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = build_norm_layer(self.norm_cfg, embedding_dim)[1]

        self.mlp = FFN(**ffn_cfg)
        self.norm3 = build_norm_layer(self.norm_cfg, embedding_dim)[1]

        self.norm4 = build_norm_layer(self.norm_cfg, embedding_dim)[1]
        self.cross_attn_image_to_token = SAMAttention(
            embedding_dim, num_heads=num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self,
                queries: Tensor,
                keys: Tensor,
                query_pe: Tensor,
                key_pe: Tensor,
                padding_mask: Tensor = None,
                prompt_attn_mask: Tensor = None,
                prompt_padding_mask: Tensor = None) -> Tuple[Tensor, Tensor]:

        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(
                queries,
                key=queries,
                value=queries,
                attn_mask=prompt_attn_mask,
                padding_mask=prompt_padding_mask
            )

        else:
            q = queries + query_pe
            attn_out = self.self_attn(
                q,
                key=q,
                value=queries,
                attn_mask=prompt_attn_mask,
                padding_mask=prompt_padding_mask
            )
            queries = queries + attn_out

        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe

        attn_out = self.cross_attn_token_to_image(
            q,
            key=k,
            value=keys,
            padding_mask=padding_mask
        )
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(
            k,
            key=q,
            value=queries,
            padding_mask=prompt_padding_mask
        )
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class SAMAttention(MultiheadAttention):
    """
    Extension of mmcv MultiheadAttention that allows for downscaling embedding
    size of q, k, and v before computing attention. Used in SAM.
    """

    def __init__(self, embed_dims, batch_first=True,
                 downsample_rate=1, **kwargs) -> None:
        orig_dim = embed_dims
        embed_dims = embed_dims // downsample_rate
        super().__init__(embed_dims, batch_first=batch_first, kdim=embed_dims+1,
                **kwargs)
        #self.attn.embed_dim = orig_dim

        # define q, k, v projections

        self.attn.q_proj_weight = nn.Parameter(torch.empty((self.embed_dims, orig_dim)))
        self.attn.k_proj_weight = nn.Parameter(torch.empty((self.embed_dims, orig_dim)))
        self.attn.v_proj_weight = nn.Parameter(torch.empty((self.embed_dims, orig_dim)))
        self.attn.out_proj = NonDynamicallyQuantizableLinear(self.embed_dims, orig_dim, bias=True)
        self.attn.in_proj_bias = nn.Parameter(torch.empty(3 * self.embed_dims))

    def forward(self,
                query,
                key=None,
                value=None,
                **kwargs):

        ## Input projections
        #query = self.q_proj(query)
        #if key is not None:
        #    key = self.k_proj(key)

        #if value is not None:
        #    value = self.v_proj(value)

        ## no skip connection in SAM attention
        #identity = torch.zeros_like(query)
        #query = super().forward(query, key=key, value=value, identity=identity, **kwargs)

        #return self.out_proj(query)

        # no skip connection in SAM attention
        identity = torch.zeros_like(query)
        return super().forward(query, key=key, value=value, identity=identity, **kwargs)


class PositionEmbeddingRandom(BaseModule):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
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
