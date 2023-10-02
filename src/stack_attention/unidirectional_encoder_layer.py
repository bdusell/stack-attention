from typing import Optional

from torch_unidirectional import Unidirectional
from .sublayer import set_tag, get_unidirectional_sublayer
from .feedforward import get_feedforward_sublayer

def get_unidirectional_encoder_layer_with_custom_attention(
    attention_func: Unidirectional,
    d_model: int,
    feedforward_size: int,
    dropout: Optional[float],
    tag: Optional[str]=None,
    main: bool=False
) -> Unidirectional:
    return (
        set_tag(get_unidirectional_sublayer(
            attention_func,
            d_model,
            dropout
        ), tag, main) |
        get_feedforward_sublayer(
            d_model,
            feedforward_size,
            dropout
        )
    )
