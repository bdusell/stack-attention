from typing import Optional

import torch

from torch_extras.embedding_layer import EmbeddingLayer
from .simple import SimpleLayerUnidirectional

class EmbeddingUnidirectional(SimpleLayerUnidirectional):

    def __init__(self,
        vocabulary_size: int,
        output_size: int,
        use_padding: bool,
        shared_embeddings: Optional[torch.Tensor]=None
    ):
        super().__init__(EmbeddingLayer(
            vocabulary_size,
            output_size,
            use_padding,
            shared_embeddings
        ))
