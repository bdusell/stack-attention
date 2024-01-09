from typing import Optional

import torch

from torch_unidirectional import (
    Unidirectional,
    ResidualUnidirectional,
    SimpleLayerUnidirectional,
    DropoutUnidirectional
)

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
