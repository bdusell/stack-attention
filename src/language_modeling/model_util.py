import functools

import humanfriendly
import torch
import torch_semiring_einsum

from lib.pytorch_tools.model_interface import ModelInterface
from torch_extras.init import smart_init, uniform_fallback
from torch_unidirectional import SimpleLayerUnidirectional, OutputUnidirectional
from transformer_model.positional_encodings import SinusoidalPositionalEncodingCacher
from transformer_model.input_layer import get_transformer_input_unidirectional
from transformer_model.unidirectional_encoder import (
    get_shared_embeddings,
    UnidirectionalTransformerEncoderLayers
)
from stack_attention.unidirectional_encoder_layer import (
    get_unidirectional_encoder_layer_with_custom_attention
)
from sequence_to_sequence.model_util import (
    parse_layers,
    get_stack_attention_func,
    pad_sequences
)

class LanguageModelingModelInterface(ModelInterface):

    def add_more_init_arguments(self, group):
        group.add_argument('--layers')
        group.add_argument('--d-model', type=int)
        group.add_argument('--num-heads', type=int)
        group.add_argument('--feedforward-size', type=int)
        group.add_argument('--dropout', type=float)
        group.add_argument('--init-scale', type=float)

    def add_forward_arguments(self, parser):
        group = parser.add_argument_group('Model execution')
        group.add_argument('--einsum-block-size', type=int)
        group.add_argument('--einsum-max-memory', type=humanfriendly.parse_size)

    def on_saver_constructed(self, args, saver):
        if args.einsum_block_size is not None:
            self.block_size = args.einsum_block_size
        else:
            self.block_size = torch_semiring_einsum.AutomaticBlockSize(
                max_cuda_bytes=args.einsum_max_memory
            )

    def get_kwargs(self, args, input_vocab_size, output_vocab_size):
        return dict(
            layers=args.layers,
            d_model=args.d_model,
            num_heads=args.num_heads,
            feedforward_size=args.feedforward_size,
            dropout=args.dropout,
            input_vocab_size=input_vocab_size,
            output_vocab_size=output_vocab_size
        )

    def construct_model(self,
        layers,
        d_model,
        num_heads,
        feedforward_size,
        dropout,
        input_vocab_size,
        output_vocab_size
    ):
        if layers is None:
            raise ValueError
        if d_model is None:
            raise ValueError
        if num_heads is None:
            raise ValueError
        if feedforward_size is None:
            raise ValueError
        if dropout is None:
            raise ValueError
        return get_unidirectional_encoder(
            input_vocabulary_size=input_vocab_size,
            output_vocabulary_size=output_vocab_size,
            layers=list(parse_layers(layers)),
            d_model=d_model,
            num_heads=num_heads,
            feedforward_size=feedforward_size,
            dropout=dropout,
            use_padding=True
        )

    def initialize(self, args, model, generator):
        if args.init_scale is None:
            raise ValueError
        smart_init(model, generator, fallback=uniform_fallback(args.init_scale))

    def adjust_length(self, length):
        # Add 1 for BOS.
        return length + 1

    def prepare_batch(self, batch, device, data):
        # Use the same index for padding symbols in both the input and output
        # tensor. The input vocab should be bigger than the output vocab, so
        # using the length of the input vocab should work fine. Using the same
        # padding symbol for both allows us to allocate one tensor and simply
        # slice it to get the input and output tensors.
        pad = len(data.input_vocab)
        whole_tensor = pad_sequences(
            batch,
            device,
            bos=data.input_vocab.bos_index,
            eos=data.output_vocab.eos_index,
            pad=pad
        )
        input_tensor = whole_tensor[:, :-1]
        output_tensor = whole_tensor[:, 1:]
        return input_tensor, output_tensor

    def get_logits(self, model, model_input):
        # Note that it is unnecessary to pass a padding mask, because padding
        # only occurs at the end of a sequence, and the model is already
        # causally masked.
        return model(
            model_input,
            include_first=False,
            tag_kwargs=dict(
                nondeterministic=dict(
                    block_size=self.block_size
                )
            )
        )

def get_unidirectional_encoder(
    input_vocabulary_size,
    output_vocabulary_size,
    layers,
    d_model,
    num_heads,
    feedforward_size,
    dropout,
    use_padding=True
):
    shared_embeddings = get_shared_embeddings(
        True,
        input_vocabulary_size,
        output_vocabulary_size,
        d_model,
        use_padding
    )
    positional_encoding_cacher = SinusoidalPositionalEncodingCacher()
    layers = get_layer_modules(
        input_vocabulary_size,
        output_vocabulary_size,
        shared_embeddings,
        positional_encoding_cacher,
        layers,
        d_model,
        num_heads,
        feedforward_size,
        dropout,
        use_padding
    )
    return functools.reduce(lambda x, y: x | y, layers)

def get_layer_modules(
    input_vocabulary_size,
    output_vocabulary_size,
    shared_embeddings,
    positional_encoding_cacher,
    layers,
    d_model,
    num_heads,
    feedforward_size,
    dropout,
    use_padding
):
    yield get_transformer_input_unidirectional(
        vocabulary_size=input_vocabulary_size,
        d_model=d_model,
        dropout=dropout,
        use_padding=use_padding,
        shared_embeddings=shared_embeddings,
        positional_encoding_cacher=positional_encoding_cacher
    )
    for module in get_middle_layer_modules(
        layers,
        d_model,
        num_heads,
        feedforward_size,
        dropout,
        use_padding
    ):
        yield module
    yield SimpleLayerUnidirectional(torch.nn.LayerNorm(d_model))
    yield OutputUnidirectional(
        input_size=d_model,
        vocabulary_size=output_vocabulary_size,
        shared_embeddings=shared_embeddings
    )

def get_middle_layer_modules(
    layers,
    d_model,
    num_heads,
    feedforward_size,
    dropout,
    use_padding
):
    for layer_type, layer_args in layers:
        if layer_type == 'transformer':
            num_layers, = layer_args
            yield UnidirectionalTransformerEncoderLayers(
                num_layers=num_layers,
                d_model=d_model,
                num_heads=num_heads,
                feedforward_size=feedforward_size,
                dropout=dropout,
                use_final_layer_norm=False,
                enable_nested_tensor=use_padding
            )
        else:
            yield get_unidirectional_encoder_layer_with_custom_attention(
                get_stack_attention_func(layer_type, layer_args, d_model),
                d_model=d_model,
                feedforward_size=feedforward_size,
                dropout=dropout,
                tag=layer_type
            )
