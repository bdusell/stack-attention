from collections.abc import Callable, Iterable
from typing import Optional, Union

import torch

from torch_unidirectional import (
    Unidirectional,
    ForwardResult,
    OutputUnidirectional
)

from .positional_encodings import SinusoidalPositionalEncodingCacher
from .input_layer import get_transformer_input_unidirectional
from .mask import make_causal_attention_mask

def get_transformer_decoder(
    input_vocabulary_size: int,
    output_vocabulary_size: int,
    shared_embeddings: Optional[torch.Tensor],
    positional_encoding_cacher: Optional[SinusoidalPositionalEncodingCacher],
    num_layers: int,
    d_model: int,
    num_heads: int,
    feedforward_size: int,
    dropout: float,
    use_padding: bool,
    tag: Optional[str]=None
) -> Unidirectional:
    return (
        get_transformer_input_unidirectional(
            vocabulary_size=input_vocabulary_size,
            d_model=d_model,
            dropout=dropout,
            use_padding=use_padding,
            shared_embeddings=shared_embeddings,
            positional_encoding_cacher=positional_encoding_cacher
        ) |
        add_tag(TransformerDecoderLayers(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            feedforward_size=feedforward_size,
            dropout=dropout,
            use_final_layer_norm=True
        ), tag) |
        OutputUnidirectional(
            input_size=d_model,
            vocabulary_size=output_vocabulary_size,
            shared_embeddings=shared_embeddings
        )
    )

def add_tag(model, tag):
    if tag is None:
        return model.main()
    else:
        return model.tag(tag)

class TransformerDecoderLayers(Unidirectional):

    def __init__(self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        feedforward_size: int,
        dropout: float,
        use_final_layer_norm: bool
    ):
        super().__init__()
        self.layers = torch.nn.TransformerDecoder(
            decoder_layer=torch.nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=feedforward_size,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_layers,
            norm=torch.nn.LayerNorm(d_model) if use_final_layer_norm else None
        )

    def forward(self,
        input_sequence: torch.Tensor,
        encoder_sequence: torch.Tensor,
        input_is_padding_mask: Optional[torch.Tensor]=None,
        encoder_is_padding_mask: Optional[torch.Tensor]=None,
        initial_state: Optional[Unidirectional.State]=None,
        return_state: bool=False,
        include_first: bool=True
    ) -> Union[torch.Tensor, ForwardResult]:
        """
        :param input_sequence: The target sequence that is given as input to
            the decoder.
        :param encoder_sequence: The output sequence of the encoder.
        :param input_is_padding_mask: A boolean tensor indicating which
            positions in the input to the decoder correspond to padding
            symbols that should be ignored. Important note: If padding only
            occurs at the end of a sequence, then providing this mask is not
            necessary, because the attention mechanism is causally masked
            anyway.
        """
        if initial_state is not None:
            # TODO
            raise NotImplementedError
        if return_state:
            # TODO
            raise NotImplementedError
        if include_first:
            raise ValueError('include_first must be False')
        return self.layers(
            tgt=input_sequence,
            memory=encoder_sequence,
            tgt_mask=make_causal_attention_mask(
                sequence_length=input_sequence.size(1),
                device=input_sequence.device,
                dtype=input_sequence.dtype
            ),
            tgt_key_padding_mask=input_is_padding_mask,
            memory_key_padding_mask=encoder_is_padding_mask
        )

    class State(Unidirectional.State):

        decoder: 'TransformerDecoderLayers'
        encoder_sequence: torch.Tensor
        encoder_is_padding_mask: torch.Tensor
        previous_inputs: torch.Tensor

        def __init__(self,
            decoder: 'TransformerDecoderLayers',
            encoder_sequence: torch.Tensor,
            encoder_is_padding_mask: torch.Tensor,
            previous_inputs: torch.Tensor
        ):
            super().__init__()
            self.decoder = decoder
            self.encoder_sequence = encoder_sequence
            self.encoder_is_padding_mask = encoder_is_padding_mask
            self.previous_inputs = previous_inputs

        def next(self, input_tensor: torch.Tensor) -> Unidirectional.State:
            return TransformerDecoderLayers.State(
                self.decoder,
                self.encoder_sequence,
                self.encoder_is_padding_mask,
                # Simply concatenate this input to the tensor of all previous
                # inputs.
                torch.concat([
                    self.previous_inputs,
                    input_tensor[:, None, :]
                ], dim=1)
            )

        def output(self) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
            # TODO This is very inefficient
            # NOTE This assumes there is no padding in the decoder input
            full_output = self.decoder.forward(
                self.previous_inputs,
                self.encoder_sequence,
                encoder_is_padding_mask=self.encoder_is_padding_mask,
                include_first=False
            )
            return full_output[:, -1]

        def batch_size(self) -> int:
            return self.previous_inputs.size(0)

        def transform_tensors(self,
            func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Unidirectional.State:
            return TransformerDecoderLayers.State(
                self.decoder,
                func(self.encoder_sequence),
                func(self.encoder_is_padding_mask),
                func(self.previous_inputs)
            )

        # TODO

        def fastforward(self, input_sequence: torch.Tensor) -> Unidirectional.State:
            raise NotImplementedError

        def states(self,
            input_sequence: torch.Tensor,
            include_first: bool
        ) -> Iterable[Unidirectional.State]:
            raise NotImplementedError

        def outputs(self,
            input_sequence: torch.Tensor,
            include_first: bool
        ) -> Union[Iterable[torch.Tensor], Iterable[tuple[torch.Tensor, ...]]]:
            raise NotImplementedError

        def forward(self,
            input_sequence: torch.Tensor,
            return_state: bool,
            include_first: bool
        ) -> Union[torch.Tensor, ForwardResult]:
            raise NotImplementedError

    def initial_state(self,
        batch_size: int,
        encoder_sequence: torch.Tensor,
        encoder_is_padding_mask: torch.Tensor
    ) -> Unidirectional.State:
        return self.State(
            self,
            encoder_sequence,
            encoder_is_padding_mask,
            torch.empty(
                (batch_size, 0, encoder_sequence.size(2)),
                dtype=encoder_sequence.dtype,
                device=encoder_sequence.device
            )
        )
