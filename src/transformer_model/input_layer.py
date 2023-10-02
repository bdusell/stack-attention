import math
from typing import Optional

import torch

from torch_extras.embedding_layer import EmbeddingLayer
from torch_unidirectional import (
    SimpleLayerUnidirectional,
    PositionalUnidirectional,
    DropoutUnidirectional
)

from .positional_encodings import SinusoidalPositionalEncodingCacher

class ScaledEmbeddingLayer(torch.nn.Module):

    def __init__(self, vocabulary_size, output_size, use_padding, shared_embeddings):
        super().__init__()
        self.embedding_layer = EmbeddingLayer(
            vocabulary_size,
            output_size,
            use_padding,
            shared_embeddings
        )
        self.register_buffer(
            'embedding_scale',
            torch.tensor(math.sqrt(output_size)),
            persistent=False
        )

    def forward(self, x):
        # Multiply the embedding weights by sqrt(d_model).
        return self.embedding_layer(x) * self.embedding_scale

class SinusoidalPositionalEncodingLayer(PositionalUnidirectional):

    def __init__(self, cacher=None):
        super().__init__()
        if cacher is None:
            cacher = SinusoidalPositionalEncodingCacher()
        self.cacher = cacher

    def forward_from_position(self, input_sequence, position):
        batch_size, sequence_length, d_model = input_sequence.size()
        positional_encodings = self.cacher.get_encodings(
            position + sequence_length,
            d_model
        )
        return input_sequence + positional_encodings[None, position:position+sequence_length]

    def forward_at_position(self, input_tensor, position):
        batch_size, d_model = input_tensor.size()
        positional_encodings = self.cacher.get_encodings(position + 1, d_model)
        positional_encoding_i = positional_encodings[position]
        return input_tensor + positional_encoding_i

def get_transformer_input_unidirectional(
    vocabulary_size: int,
    d_model: int,
    dropout: Optional[float],
    use_padding: bool,
    shared_embeddings: Optional[torch.nn.Parameter]=None,
    positional_encoding_cacher: Optional[SinusoidalPositionalEncodingCacher]=None
):
    # Apply the following layers in this order:
    # 1. scaled embedding layer
    # 2. sinusoidal positional encoding layer
    # 3. dropout layer (optional)
    result = (
        SimpleLayerUnidirectional(ScaledEmbeddingLayer(
            vocabulary_size=vocabulary_size,
            output_size=d_model,
            use_padding=use_padding,
            shared_embeddings=shared_embeddings
        )) |
        SinusoidalPositionalEncodingLayer(
            positional_encoding_cacher
        )
    )
    if dropout:
        result = result | DropoutUnidirectional(dropout)
    return result
