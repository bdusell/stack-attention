from typing import Optional, Union

import torch

from torch_unidirectional import (
    Unidirectional,
    ForwardResult,
    SimpleReshapingLayerUnidirectional
)

from .unidirectional_encoder_layer import (
    get_unidirectional_sublayer,
    get_feedforward_sublayer,
    set_tag
)

def get_decoder_layer_with_custom_attention(
    attention_func: Unidirectional,
    d_model: int,
    feedforward_size: int,
    dropout: Optional[float],
    num_cross_attention_heads: int,
    tag: Optional[str]=None,
    main: bool=False,
    cross_attention_tag: str='cross_attention'
) -> Unidirectional:
    return (
        set_tag(get_unidirectional_sublayer(
            attention_func,
            d_model,
            dropout
        ), tag, main) |
        get_unidirectional_sublayer(
            CrossAttentionUnidirectional(
                d_model,
                num_cross_attention_heads,
                dropout
            ),
            d_model,
            dropout
        ).tag(cross_attention_tag) |
        get_feedforward_sublayer(
            d_model,
            feedforward_size,
            dropout
        )
    )

class CrossAttention(torch.nn.Module):

    def __init__(self,
        d_model: int,
        num_heads: int,
        dropout: float
    ):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(self,
        input_sequence: torch.Tensor,
        encoder_sequence: torch.Tensor,
        encoder_is_padding_mask: Optional[torch.Tensor]=None
    ):
        """
        :param input_sequence: The target sequence that is given as input to
            the cross-attention sublayer.
        :param encoder_sequence: The output sequence of the encoder.
        """
        return self.attention(
            input_sequence,
            encoder_sequence,
            encoder_sequence,
            key_padding_mask=encoder_is_padding_mask,
            need_weights=False
        )[0]

class CrossAttentionUnidirectional(SimpleReshapingLayerUnidirectional):

    def __init__(self,
        d_model: int,
        num_heads: int,
        dropout: float
    ):
        super().__init__(CrossAttention(d_model, num_heads, dropout))

    def transform_kwargs(self, kwargs, func):
        kwargs = kwargs.copy()
        kwargs['encoder_sequence'] = func(kwargs['encoder_sequence'])
        kwargs['encoder_is_padding_mask'] = func(kwargs['encoder_is_padding_mask'])
        return kwargs
