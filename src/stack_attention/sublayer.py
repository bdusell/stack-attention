from typing import Optional

import torch

from torch_unidirectional import (
    Unidirectional,
    ResidualUnidirectional,
    SimpleLayerUnidirectional,
    DropoutUnidirectional
)

class Sublayer:
    """A sublayer of a transformer encoder. Adds layer norm, dropout, and
    residual connections to a sublayer function."""

    def __init__(self,
        sublayer_func: torch.nn.Module,
        d_model: int,
        dropout: Optional[float]
    ):
        super().__init__()
        self.sublayer_func = sublayer_func
        self.layer_norm = torch.nn.LayerNorm((d_model,))
        self.dropout = torch.nn.Dropout(dropout) if dropout else torch.nn.Identity()

    def forward(self, input_sequence, *args, **kwargs):
        # input_sequence : batch_size x sequence_length x d_model
        # return : batch_size x sequence_length x d_model
        return input_sequence + self.dropout(self.sublayer_func(self.layer_norm(input_sequence), *args, **kwargs))

def get_unidirectional_sublayer(
    sublayer_func: Unidirectional,
    d_model: int,
    dropout: Optional[float]
) -> Unidirectional:
    return ResidualUnidirectional(
        SimpleLayerUnidirectional(torch.nn.LayerNorm((d_model,))) |
        sublayer_func.main() |
        DropoutUnidirectional(dropout)
    )

def set_tag(model, tag, main):
    if tag is not None:
        model = model.tag(tag)
    if main:
        model = model.main()
    return model
