from typing import Dict, List, Tuple, Union, Optional

from torch import nn, Tensor
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmdet.utils import OptMultiConfig

@MODELS.register_module()
class LabelEmbedEncoder(BaseModule):
    """
    Encodes labels into embedding space.
    """

    def __init__(self, num_classes: int = 5, embed_dims: int = 256,
                 init_cfg: OptMultiConfig = None) -> None:
        """
        Initialize the LabelEncoder.

        Args:
            num_classes (int): Number of classes. Default: 5.
              for pos, neg, box_corner_a, box_corner_b, not_a_point
            embed_dims (int): Dimension of the embedding. Default: 256.
        """
        super().__init__(init_cfg)
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize the embedding layer."""
        self.label_embedding = nn.Embedding(self.num_classes, self.embed_dims)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        nn.init.uniform_(self.label_embedding.weight)

    def forward(self, labels: Tensor) -> Tensor:
        """
        Forward pass of the LabelEncoder.

        Args:
            labels (Tensor): Tensor of labels.
                shape: (bs, num_instance*num_points)

        Returns:
            Tensor: The embedding of the labels.
        """
        return self.label_embedding(labels)
