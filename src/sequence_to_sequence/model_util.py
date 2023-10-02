import dataclasses
import functools
import re

import humanfriendly
import torch
import torch_semiring_einsum

from lib.pytorch_tools.model_interface import ModelInterface
from torch_extras.init import smart_init, uniform_fallback
from torch_extras.compose import Composable
from torch_unidirectional import SimpleLayerUnidirectional, OutputUnidirectional
from transformer_model.positional_encodings import SinusoidalPositionalEncodingCacher
from transformer_model.input_layer import get_transformer_input_unidirectional
from transformer_model.encoder_decoder import get_shared_embeddings
from transformer_model.encoder import get_transformer_encoder, TransformerEncoderLayers
from transformer_model.decoder import get_transformer_decoder, TransformerDecoderLayers
from stack_attention.unidirectional_encoder_layer import (
    get_unidirectional_encoder_layer_with_custom_attention
)
from stack_attention.decoder_layer import (
    get_decoder_layer_with_custom_attention
)
from stack_attention.superposition import SuperpositionStackAttention
from stack_attention.nondeterministic import NondeterministicStackAttention
from .beam_search import beam_search

class SequenceToSequenceModelInterface(ModelInterface):

    def add_more_init_arguments(self, group):
        group.add_argument('--encoder-layers')
        group.add_argument('--use-standard-encoder', action='store_true', default=False)
        group.add_argument('--decoder-layers')
        group.add_argument('--use-standard-decoder', action='store_true', default=False)
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
        if hasattr(args, 'einsum_block_size'):
            if args.einsum_block_size is not None:
                self.block_size = args.einsum_block_size
            else:
                self.block_size = torch_semiring_einsum.AutomaticBlockSize(
                    max_cuda_bytes=args.einsum_max_memory
                )

    def get_kwargs(self, args, source_vocab_size, target_input_vocab_size,
            target_output_vocab_size, tie_embeddings):
        return dict(
            encoder_layers=args.encoder_layers,
            use_standard_encoder=args.use_standard_encoder,
            decoder_layers=args.decoder_layers,
            use_standard_decoder=args.use_standard_decoder,
            d_model=args.d_model,
            num_heads=args.num_heads,
            feedforward_size=args.feedforward_size,
            dropout=args.dropout,
            source_vocab_size=source_vocab_size,
            target_input_vocab_size=target_input_vocab_size,
            target_output_vocab_size=target_output_vocab_size,
            tie_embeddings=tie_embeddings
        )

    def construct_model(self,
        encoder_layers,
        use_standard_encoder,
        decoder_layers,
        use_standard_decoder,
        d_model,
        num_heads,
        feedforward_size,
        dropout,
        source_vocab_size,
        target_input_vocab_size,
        target_output_vocab_size,
        tie_embeddings
    ):
        if encoder_layers is None:
            raise ValueError
        if decoder_layers is None:
            raise ValueError
        if d_model is None:
            raise ValueError
        if num_heads is None:
            raise ValueError
        if feedforward_size is None:
            raise ValueError
        if dropout is None:
            raise ValueError
        return get_encoder_decoder(
            source_vocabulary_size=source_vocab_size,
            target_input_vocabulary_size=target_input_vocab_size,
            target_output_vocabulary_size=target_output_vocab_size,
            tie_embeddings=tie_embeddings,
            encoder_layers=list(parse_layers(encoder_layers)),
            use_standard_encoder=use_standard_encoder,
            decoder_layers=list(parse_layers(decoder_layers)),
            use_standard_decoder=use_standard_decoder,
            d_model=d_model,
            num_heads=num_heads,
            feedforward_size=feedforward_size,
            dropout=dropout,
            use_source_padding=True,
            use_target_padding=True
        )

    def initialize(self, args, model, generator):
        if args.init_scale is None:
            raise ValueError
        smart_init(model, generator, fallback=uniform_fallback(args.init_scale))

    def adjust_source_length(self, source_length):
        # Add 1 for EOS.
        return source_length + 1

    def adjust_target_length(self, target_length):
        # Add 1 for BOS.
        return target_length + 1

    def get_space_polynomial_terms(self, batch_size, source_length, target_length):
        without_batch_size = [
            source_length ** 2,
            source_length,
            source_length * target_length,
            target_length ** 2,
            target_length,
            1
        ]
        with_batch_size = [batch_size * x for x in without_batch_size]
        return without_batch_size + with_batch_size

    def prepare_batch(self, batch, device, data):
        model_source = self.prepare_source([s for s, t in batch], device, data)
        target_input_pad = len(data.target_input_vocab)
        target_input = pad_sequences(
            [t for s, t in batch],
            device,
            bos=data.target_input_vocab.bos_index,
            pad=target_input_pad
        )
        target_output_pad = len(data.target_output_vocab)
        target_output = pad_sequences(
            [t for s, t in batch],
            device,
            eos=data.target_output_vocab.eos_index,
            pad=target_output_pad
        )
        model_input = ModelSourceAndTarget(
            source=model_source.source,
            source_is_padding_mask=model_source.source_is_padding_mask,
            target=target_input
        )
        return model_input, target_output

    def prepare_source(self, sources, device, data):
        source_pad = len(data.source_vocab)
        source = pad_sequences(
            sources,
            device,
            eos=data.source_vocab.eos_index,
            pad=source_pad
        )
        return ModelSource(
            source=source,
            source_is_padding_mask=(source == source_pad)
        )

    def on_before_process_pairs(self, saver, datasets):
        max_length = max(
            max(
                self.adjust_source_length(len(s)),
                self.adjust_target_length(len(t))
            )
            for dataset in datasets
            for s, t in dataset
        )
        return self._preallocate_positional_encodings(saver, max_length)

    def on_before_decode(self, saver, datasets, max_target_length):
        data_max_length = max(
            self.adjust_source_length(len(s))
            for dataset in datasets
            for s in dataset
        )
        # Subtract 1 because beam search doesn't need the last input.
        max_target_length = self.adjust_target_length(max_target_length) - 1
        max_length = max(max_target_length, data_max_length)
        return self._preallocate_positional_encodings(saver, max_length)

    def _preallocate_positional_encodings(self, saver, max_length):
        d_model = saver.kwargs['d_model']
        for module in saver.model.modules():
            if isinstance(module, SinusoidalPositionalEncodingCacher):
                module.get_encodings(max_length, d_model)
                module.set_allow_reallocation(False)

    def get_logits(self, model, model_input):
        # Note that it is unnecessary to pass a padding mask for the target
        # side, because padding only occurs at the end of a sequence, and the
        # decoder is already causally masked.
        return model(
            source_sequence=model_input.source,
            target_sequence=model_input.target,
            encoder_kwargs=self.get_encoder_kwargs(model_input),
            decoder_kwargs=self.get_decoder_kwargs(model_input)
        )

    def decode(self, model, model_source, bos_symbol, beam_size, eos_symbol, max_length):
        model.eval()
        with torch.no_grad():
            decoder_state = model.initial_decoder_state(
                source_sequence=model_source.source,
                encoder_kwargs=self.get_encoder_kwargs(model_source),
                decoder_kwargs=self.get_decoder_kwargs(model_source)
            )
            device = model_source.source.device
            # Feed BOS into the model at the first timestep.
            decoder_state = decoder_state.next(torch.full(
                (decoder_state.batch_size(),),
                bos_symbol,
                dtype=torch.long,
                device=device
            ))
            return beam_search(decoder_state, beam_size, eos_symbol, max_length, device)

    def get_encoder_kwargs(self, model_source):
        return dict(tag_kwargs=dict(
            transformer=dict(
                is_padding_mask=model_source.source_is_padding_mask
            ),
            nondeterministic=dict(
                block_size=self.block_size
            )
        ))

    def get_decoder_kwargs(self, model_source):
        return dict(tag_kwargs=dict(
            transformer=dict(
                encoder_is_padding_mask=model_source.source_is_padding_mask
            ),
            nondeterministic=dict(
                block_size=self.block_size
            )
        ))

@dataclasses.dataclass
class ModelSource:
    source: torch.Tensor
    source_is_padding_mask: torch.Tensor

@dataclasses.dataclass
class ModelSourceAndTarget(ModelSource):
    target: torch.Tensor

def pad_sequences(sequences, device, pad, bos=None, eos=None):
    batch_size = len(sequences)
    max_length = max(map(len, sequences))
    add_bos = bos is not None
    bos_offset = int(add_bos)
    add_eos = eos is not None
    sequence_length = bos_offset + max_length + int(add_eos)
    result = torch.full(
        (batch_size, sequence_length),
        pad,
        dtype=torch.long,
        device=device
    )
    if add_bos:
        result[:, 0] = bos
    for i, sequence in enumerate(sequences):
        result[i, bos_offset:bos_offset+len(sequence)] = sequence
        if add_eos:
            result[i, bos_offset+len(sequence)] = eos
    return result

LAYER_RE = re.compile(r'^(?:(\d+)|superposition-(\d+)|nondeterministic-(\d+)-(\d+)-(\d+))$')

def parse_layers(s):
    for part in s.split('.'):
        m = LAYER_RE.match(part)
        if m is None:
            raise ValueError(f'layer type not recognized: {part!r}')
        (
            trans_num_layers,
            sup_vector_size,
            nondet_num_states,
            nondet_stack_alphabet_size,
            nondet_vector_size
        ) = m.groups()
        if trans_num_layers is not None:
            yield 'transformer', (int(trans_num_layers),)
        elif sup_vector_size is not None:
            yield 'superposition', (int(sup_vector_size),)
        else:
            yield 'nondeterministic', (int(nondet_num_states), int(nondet_stack_alphabet_size), int(nondet_vector_size))

def get_stack_attention_func(layer_type, layer_args, d_model):
    if layer_type == 'superposition':
        stack_embedding_size, = layer_args
        return SuperpositionStackAttention(
            d_model=d_model,
            stack_embedding_size=stack_embedding_size
        )
    elif layer_type == 'nondeterministic':
        num_states, stack_alphabet_size, stack_embedding_size = layer_args
        return NondeterministicStackAttention(
            d_model=d_model,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            stack_embedding_size=stack_embedding_size
        )
    else:
        raise NotImplementedError

def get_encoder_decoder(
    source_vocabulary_size,
    target_input_vocabulary_size,
    target_output_vocabulary_size,
    tie_embeddings,
    encoder_layers,
    use_standard_encoder,
    use_standard_decoder,
    decoder_layers,
    d_model,
    num_heads,
    feedforward_size,
    dropout,
    use_source_padding=True,
    use_target_padding=True
):
    shared_embeddings = get_shared_embeddings(
        tie_embeddings,
        source_vocabulary_size,
        target_input_vocabulary_size,
        target_output_vocabulary_size,
        d_model,
        use_source_padding,
        use_target_padding
    )
    positional_encoding_cacher = SinusoidalPositionalEncodingCacher()
    return EncoderDecoder(
        get_encoder(
            vocabulary_size=source_vocabulary_size,
            shared_embeddings=shared_embeddings,
            positional_encoding_cacher=positional_encoding_cacher,
            layers=encoder_layers,
            use_standard_encoder=use_standard_encoder,
            d_model=d_model,
            num_heads=num_heads,
            feedforward_size=feedforward_size,
            dropout=dropout,
            use_padding=use_source_padding
        ),
        get_decoder(
            input_vocabulary_size=target_input_vocabulary_size,
            output_vocabulary_size=target_output_vocabulary_size,
            shared_embeddings=shared_embeddings,
            positional_encoding_cacher=positional_encoding_cacher,
            layers=decoder_layers,
            use_standard_decoder=use_standard_decoder,
            d_model=d_model,
            num_heads=num_heads,
            feedforward_size=feedforward_size,
            dropout=dropout,
            use_padding=use_target_padding
        )
    )

def get_encoder(
    vocabulary_size,
    shared_embeddings,
    positional_encoding_cacher,
    layers,
    use_standard_encoder,
    d_model,
    num_heads,
    feedforward_size,
    dropout,
    use_padding
):
    if use_standard_encoder:
        for layer_type, _ in layers:
            if layer_type != 'transformer':
                raise ValueError(f'cannot have layer type {layer_type} with standard encoder')
        num_encoder_layers = sum(x for _, (x,) in layers)
        return get_transformer_encoder(
            vocabulary_size,
            shared_embeddings,
            positional_encoding_cacher,
            num_encoder_layers,
            d_model,
            num_heads,
            feedforward_size,
            dropout,
            use_padding,
            tag='transformer'
        )
    layers = get_encoder_layer_modules(
        vocabulary_size,
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

def get_encoder_layer_modules(
    vocabulary_size,
    shared_embeddings,
    positional_encoding_cacher,
    layers,
    d_model,
    num_heads,
    feedforward_size,
    dropout,
    use_padding
):
    yield Composable(
        get_transformer_input_unidirectional(
            vocabulary_size=vocabulary_size,
            d_model=d_model,
            dropout=dropout,
            use_padding=use_padding,
            shared_embeddings=shared_embeddings,
            positional_encoding_cacher=positional_encoding_cacher
        )
    ).kwargs(include_first=False)
    for module in get_encoder_middle_layer_modules(
        layers,
        d_model,
        num_heads,
        feedforward_size,
        dropout,
        use_padding
    ):
        yield module
    yield Composable(torch.nn.LayerNorm(d_model))

def get_encoder_middle_layer_modules(
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
            yield Composable(TransformerEncoderLayers(
                num_layers=num_layers,
                d_model=d_model,
                num_heads=num_heads,
                feedforward_size=feedforward_size,
                dropout=dropout,
                use_final_layer_norm=False,
                enable_nested_tensor=use_padding
            )).tag(layer_type)
        else:
            yield Composable(get_unidirectional_encoder_layer_with_custom_attention(
                get_stack_attention_func(layer_type, layer_args, d_model),
                d_model=d_model,
                feedforward_size=feedforward_size,
                dropout=dropout,
                main=True
            )).kwargs(include_first=False).tag(layer_type)

def get_decoder(
    input_vocabulary_size,
    output_vocabulary_size,
    shared_embeddings,
    positional_encoding_cacher,
    layers,
    use_standard_decoder,
    d_model,
    num_heads,
    feedforward_size,
    dropout,
    use_padding
):
    if use_standard_decoder:
        for layer_type, _ in layers:
            if layer_type != 'transformer':
                raise ValueError(f'cannot have layer type {layer_type} with standard encoder')
        num_decoder_layers = sum(x for _, (x,) in layers)
        return get_transformer_decoder(
            input_vocabulary_size,
            output_vocabulary_size,
            shared_embeddings,
            positional_encoding_cacher,
            num_decoder_layers,
            d_model,
            num_heads,
            feedforward_size,
            dropout,
            use_padding,
            tag='transformer'
        )
    layers = get_decoder_layer_modules(
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

def get_decoder_layer_modules(
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
    for module in get_decoder_middle_layer_modules(
        layers,
        d_model,
        num_heads,
        feedforward_size,
        dropout
    ):
        yield module
    yield SimpleLayerUnidirectional(torch.nn.LayerNorm(d_model))
    yield OutputUnidirectional(
        input_size=d_model,
        vocabulary_size=output_vocabulary_size,
        shared_embeddings=shared_embeddings
    )

def get_decoder_middle_layer_modules(
    layers,
    d_model,
    num_heads,
    feedforward_size,
    dropout
):
    for layer_type, layer_args in layers:
        if layer_type == 'transformer':
            num_layers, = layer_args
            yield TransformerDecoderLayers(
                num_layers=num_layers,
                d_model=d_model,
                num_heads=num_heads,
                feedforward_size=feedforward_size,
                dropout=dropout,
                use_final_layer_norm=False
            ).tag(layer_type)
        else:
            yield get_decoder_layer_with_custom_attention(
                get_stack_attention_func(layer_type, layer_args, d_model),
                d_model=d_model,
                feedforward_size=feedforward_size,
                dropout=dropout,
                num_cross_attention_heads=num_heads,
                tag=layer_type,
                cross_attention_tag='transformer'
            )

class EncoderDecoder(torch.nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,
        source_sequence,
        target_sequence,
        encoder_kwargs,
        decoder_kwargs
    ):
        encoder_outputs = self.encoder(
            source_sequence,
            **encoder_kwargs
        )
        decoder_kwargs['tag_kwargs']['transformer']['encoder_sequence'] = encoder_outputs
        return self.decoder(
            target_sequence,
            **decoder_kwargs,
            include_first=False
        )

    def initial_decoder_state(self, source_sequence, encoder_kwargs, decoder_kwargs):
        encoder_outputs = self.encoder(
            source_sequence,
            **encoder_kwargs
        )
        decoder_kwargs['tag_kwargs']['transformer']['encoder_sequence'] = encoder_outputs
        return self.decoder.initial_state(
            batch_size=encoder_outputs.size(0),
            **decoder_kwargs
        )
