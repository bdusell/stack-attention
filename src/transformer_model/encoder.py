from typing import Optional

import torch

from torch_extras.compose import Composable

from .positional_encodings import SinusoidalPositionalEncodingCacher
from .input_layer import get_transformer_input_unidirectional

def get_transformer_encoder(
    vocabulary_size: int,
    shared_embeddings: Optional[torch.Tensor],
    positional_encoding_cacher: Optional[SinusoidalPositionalEncodingCacher],
    num_layers: int,
    d_model: int,
    num_heads: int,
    feedforward_size: int,
    dropout: float,
    use_padding: bool,
    tag: Optional[str]=None
):
    return (
        Composable(
            get_transformer_input_unidirectional(
                vocabulary_size,
                d_model,
                dropout,
                use_padding,
                shared_embeddings,
                positional_encoding_cacher
            )
        ).kwargs(include_first=False) |
        add_tag(Composable(
            TransformerEncoderLayers(
                num_layers,
                d_model,
                num_heads,
                feedforward_size,
                dropout,
                use_final_layer_norm=True,
                enable_nested_tensor=use_padding
            )
        ), tag)
    )

def add_tag(model, tag):
    if tag is None:
        return model.main()
    else:
        return model.tag(tag)

class TransformerEncoderLayers(torch.nn.Module):

    def __init__(self,
        num_layers,
        d_model,
        num_heads,
        feedforward_size,
        dropout,
        use_final_layer_norm,
        enable_nested_tensor
    ):
        super().__init__()
        self.layers = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=feedforward_size,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_layers,
            norm=torch.nn.LayerNorm(d_model) if use_final_layer_norm else None,
            enable_nested_tensor=enable_nested_tensor
        )

    def forward(self, source_sequence, is_padding_mask=None):
        return self.layers(source_sequence, src_key_padding_mask=is_padding_mask)
