import functools
import re

import torch
import torch_semiring_einsum

from torch_extras.init import smart_init, uniform_fallback
from torch_unidirectional import (
    SimpleLayerUnidirectional,
    OutputUnidirectional,
    UnidirectionalLSTM
)
from lib.pretty_table import align, green, red
from lib.pytorch_tools.model_interface import ModelInterface
from stack_rnn_models.grefenstette import GrefenstetteRNN
from stack_rnn_models.joulin_mikolov import JoulinMikolovRNN
from stack_rnn_models.nondeterministic_stack import NondeterministicStackRNN
from stack_rnn_models.vector_nondeterministic_stack import VectorNondeterministicStackRNN
from transformer_model.unidirectional_encoder import UnidirectionalTransformerEncoderLayers
from transformer_model.input_layer import get_transformer_input_unidirectional
from stack_attention.unidirectional_encoder_layer import (
    get_unidirectional_encoder_layer_with_custom_attention
)
from stack_attention.superposition import SuperpositionStackAttention
from stack_attention.nondeterministic import NondeterministicStackAttention

class CFLModelInterface(ModelInterface):

    def add_more_init_arguments(self, group):
        group.add_argument('--model-type', choices=[
            'lstm',
            'gref',
            'jm',
            'ns',
            'vns',
            'transformer'
        ], required=True,
            help='The type of model to use. Choices are "lstm" (LSTM), "gref" '
                 '(Grefenstette et al. 2015), "jm" (Joulin & Mikolov 2015), '
                 'and "ns" (Nondeterministic Stack RNN).')
        group.add_argument('--hidden-units', type=int,
            default=20,
            help='The number of hidden units used in the LSTM controller.')
        group.add_argument('--layers', type=int,
            default=1,
            help='The number of layers used in the LSTM controller.')
        group.add_argument('--stack-embedding-size', type=int, nargs='+',
            help='(gref and jm only) The size of the embeddings used in the '
                 'differentiable stack.')
        group.add_argument('--push-hidden-state', action='store_true', default=False,
            help='(jm only) Use the hidden state as the pushed vector.')
        group.add_argument('--num-states', type=int,
            help='(ns only) The number of PDA states used by the NS-RNN.')
        group.add_argument('--stack-alphabet-size', type=int,
            help='(ns only) The number of symbols in the stack alphabet used '
                 'by the NS-RNN.')
        group.add_argument('--normalize-operations', action='store_true', default=False,
            help='(ns only) Normalize the stack operation weights so that '
                 'they sum to one.')
        group.add_argument('--no-states-in-reading', action='store_true', default=False,
            help='(ns only) Do not include PDA states in the stack reading.')
        group.add_argument('--original-bottom-symbol-behavior', action='store_true', default=False,
            help='(ns and vns only) Use the original behavior that prevents the '
                 'initial bottom symbol from being uncovered by a pop '
                 'operation once a symbol has been pushed on top of it.')
        group.add_argument('--bottom-vector', choices=['learned', 'zero', 'one'],
            default='learned',
            help='(vns only) What kind of vector to use as the bottom vector '
                 'of the stack.')
        group.add_argument('--use-stack-reading-layers', action='store_true', default=False,
            help='(ns only) Add two hidden layers between the stack reading '
                 'and the controller.')
        group.add_argument('--include-reading-in-output', action='store_true', default=False,
            help='(ns only) Concatenate the stack reading with the hidden '
                 'state when computing the output of the RNN, allowing it to '
                 'read a symbol, execute stack actions, and make predictions '
                 'in the same timestep.')
        group.add_argument('--no-normalize-reading', action='store_true', default=False,
            help='(ns only) Do not normalize the weights in the stack '
                 'reading, but apply a tanh to the log of the weights instead.')
        group.add_argument('--transformer-layers',
            help='(transformer only) A list of layers to use in the '
                 'transformer encoder. An integer n indicates a stack of n '
                 'standard attention and feedforward layers. Other choices '
                 'are jm and vrns.')
        group.add_argument('--d-model', type=int,
            help='(transformer only) Size of the hidden representations in '
                 'the transformer.')
        group.add_argument('--num-heads', type=int,
            help='(transformer only) Number of attention heads in attention '
                 'sub-layers.')
        group.add_argument('--feedforward-size', type=int,
            help='(transformer only) Size of the hidden layer in the '
                 'feedforward sub-layers.')
        group.add_argument('--transformer-sublayer-dropout', type=float,
            help='(transformer only) Dropout rate applied to every sublayer '
                 'of the transformer.')
        group.add_argument('--init-scale', type=float, default=0.1,
            help='Scales the interval from which parameters are initialized '
                 'uniformly (fully-connected layers outside the LSTM ignore '
                 'this and always use Xavier initialization).')

    def add_forward_arguments(self, parser):
        group = parser.add_argument_group('Forward pass options')
        group.add_argument('--block-size', type=int,
            default=torch_semiring_einsum.AUTOMATIC_BLOCK_SIZE,
            help='(ns only) The block size used in einsum operations. Default is automatic.')

    def get_kwargs(self, args, input_size, output_size):
        if args.model_type == 'jm':
            stack_embedding_size = None
            stack_embedding_sizes = args.stack_embedding_size
        else:
            if args.stack_embedding_size is not None:
                if len(args.stack_embedding_size) > 1:
                    raise ValueError(
                        f'--stack-embedding-size cannot have more than one value '
                        f'for --model-type {args.model_type}')
                stack_embedding_size, = args.stack_embedding_size
                stack_embedding_sizes = None
            else:
                stack_embedding_size = stack_embedding_sizes = None
        if args.model_type == 'transformer':
            transformer_layers = list(parse_transformer_layers(args.transformer_layers))
            if not transformer_layers:
                raise ValueError(
                    '--transformer-layers is required to specify at least '
                    'one layer'
                )
        else:
            transformer_layers = None
        return dict(
            model_type=args.model_type,
            input_size=input_size,
            hidden_units=args.hidden_units,
            layers=args.layers,
            output_size=output_size,
            stack_embedding_size=stack_embedding_size,
            stack_embedding_sizes=stack_embedding_sizes,
            push_hidden_state=args.push_hidden_state,
            num_states=args.num_states,
            stack_alphabet_size=args.stack_alphabet_size,
            normalize_operations=args.normalize_operations,
            include_states_in_reading=not args.no_states_in_reading,
            original_bottom_symbol_behavior=args.original_bottom_symbol_behavior,
            use_stack_reading_layers=args.use_stack_reading_layers,
            include_reading_in_output=args.include_reading_in_output,
            normalize_reading=not args.no_normalize_reading,
            bottom_vector=args.bottom_vector,
            transformer_layers=transformer_layers,
            d_model=args.d_model,
            num_heads=args.num_heads,
            feedforward_size=args.feedforward_size,
            transformer_sublayer_dropout=args.transformer_sublayer_dropout
        )

    def construct_model(self,
        model_type,
        input_size,
        hidden_units,
        output_size,
        layers,
        stack_embedding_size,
        num_states,
        stack_alphabet_size,
        normalize_operations,
        include_states_in_reading,
        transformer_layers=None,
        d_model=None,
        num_heads=None,
        feedforward_size=None,
        transformer_sublayer_dropout=None,
        original_bottom_symbol_behavior=False,
        use_stack_reading_layers=False,
        include_reading_in_output=False,
        normalize_reading=True,
        bottom_vector='zero',
        stack_embedding_sizes=None,
        push_hidden_state=False
    ):

        def construct_controller(input_size):
            return UnidirectionalLSTM(
                input_size=input_size,
                hidden_units=hidden_units,
                layers=layers
            )

        if model_type == 'lstm':
            rnn = construct_controller(input_size)
        elif model_type == 'gref':
            if stack_embedding_size is None:
                raise ValueError('--stack-embedding-size is required')
            rnn = GrefenstetteRNN(
                input_size=input_size,
                stack_embedding_size=stack_embedding_size,
                controller=construct_controller,
                controller_output_size=hidden_units
            )
        elif model_type == 'jm':
            if push_hidden_state:
                if stack_embedding_sizes is not None:
                    raise ValueError('do not use --stack-embedding-size with --push-hidden-state')
                stack_embedding_sizes = hidden_units
            else:
                if stack_embedding_sizes is None:
                    # Handles old models.
                    stack_embedding_sizes = [stack_embedding_size]
                if stack_embedding_sizes is None:
                    raise ValueError('--stack-embedding-size is required')
            rnn = JoulinMikolovRNN(
                input_size=input_size,
                stack_embedding_size=stack_embedding_sizes,
                controller=construct_controller,
                controller_output_size=hidden_units,
                push_hidden_state=push_hidden_state
            )
        elif model_type == 'ns':
            if num_states is None or stack_alphabet_size is None:
                raise ValueError('--num-states and --stack-alphabet-size are required')
            if use_stack_reading_layers:
                reading_layer_sizes = [None, None]
            else:
                reading_layer_sizes = None
            rnn = NondeterministicStackRNN(
                input_size=input_size,
                num_states=num_states,
                stack_alphabet_size=stack_alphabet_size,
                controller=construct_controller,
                controller_output_size=hidden_units,
                normalize_operations=normalize_operations,
                include_states_in_reading=include_states_in_reading,
                normalize_reading=normalize_reading,
                original_bottom_symbol_behavior=original_bottom_symbol_behavior,
                reading_layer_sizes=reading_layer_sizes,
                include_reading_in_output=include_reading_in_output
            ).tag('nondeterministic')
        elif model_type == 'vns':
            if num_states is None or stack_alphabet_size is None:
                raise ValueError('--num-states and --stack-alphabet-size are required')
            rnn = VectorNondeterministicStackRNN(
                input_size=input_size,
                num_states=num_states,
                stack_alphabet_size=stack_alphabet_size,
                stack_embedding_size=stack_embedding_size,
                controller=construct_controller,
                controller_output_size=hidden_units,
                normalize_operations=normalize_operations,
                original_bottom_symbol_behavior=original_bottom_symbol_behavior,
                bottom_vector=bottom_vector
            ).tag('nondeterministic')
        elif model_type == 'transformer':
            if (
                d_model is None or
                num_heads is None or
                feedforward_size is None or
                transformer_sublayer_dropout is None
            ):
                raise ValueError(
                    '--d-model, --num-heads, --feedforward-size, and '
                    '--transformer-sublayer-dropout are required'
                )
            layers = get_transformer_layer_modules(
                transformer_layers,
                # Add 1 to the size of the embedding table to account for BOS.
                input_size=input_size + 1,
                output_size=output_size,
                d_model=d_model,
                num_heads=num_heads,
                feedforward_size=feedforward_size,
                dropout=transformer_sublayer_dropout
            )
            return functools.reduce(lambda x, y: x | y, layers)
        else:
            raise ValueError

        return (
            rnn.main() |
            OutputUnidirectional(
                input_size=hidden_units,
                vocabulary_size=output_size
            )
        )

    def initialize(self, args, model, generator):
        smart_init(model, generator, fallback=uniform_fallback(args.init_scale))

    def on_saver_constructed(self, args, saver):
        model_type = saver.kwargs['model_type']
        if model_type == 'transformer':
            self._to_input_tensor = self._to_int_input_tensor
        else:
            self._to_input_tensor = self._to_one_hot_input_tensor
        if hasattr(args, 'block_size'):
            if model_type == 'transformer':
                self._forward_kwargs = dict(
                    include_first=False,
                    tag_kwargs=dict(
                        nondeterministic=dict(
                            block_size=args.block_size
                        )
                    )
                )
            else:
                self._forward_kwargs = dict(
                    include_first=True,
                    tag_kwargs=dict(
                        nondeterministic=dict(
                            block_size=args.block_size
                        )
                    )
                )
        self.model_input_size = saver.kwargs['input_size']

    def convert_data_to_tensors(self, data, device):
        input_vocab_size = data.input_vocab.size()
        eos_symbol = data.output_vocab.end_symbol
        data.train_data = self.batches_to_tensors(data.train_data, device, input_vocab_size, eos_symbol)
        data.valid_data = self.batches_to_tensors(data.valid_data, device, input_vocab_size, eos_symbol)

    def batches_to_tensors(self, batches, device, input_vocab_size, eos_symbol):
        return [
            self.batch_to_tensor_pair(batch, device, input_vocab_size, eos_symbol)
            for batch in batches
        ]

    def batch_to_tensor_pair(self, batch, device, input_vocab_size, eos_symbol):
        # Append EOS to the elements in the output.
        # The output includes all elements of the sequence, including the first,
        # because we do want to assign a probability to the first element of the
        # sequence.
        y = torch.tensor([element + [eos_symbol] for element in batch], device=device, dtype=torch.long)
        x = self._to_input_tensor(y, device, input_vocab_size)
        return x, y

    def _to_int_input_tensor(self, y, device, input_vocab_size):
        # Add BOS to the beginning of each element, and remove EOS from the end
        # of each element.
        bos = input_vocab_size
        # bos_tensor : batch_size x 1
        bos_tensor = torch.tensor(bos, device=device, dtype=torch.long)[None, None].expand(y.size(0), 1)
        return torch.concat([bos_tensor, y[:, :-1]], dim=1)

    def _to_one_hot_input_tensor(self, y, device, input_vocab_size):
        # Remove EOS from the end of each element and convert to one-hot
        # vectors.
        # If the input size is off by one, this means we are trying to load an
        # old model that was trained when the input vocab included an
        # unnecessary EOS symbol.
        if input_vocab_size == self.model_input_size - 1:
            input_vocab_size += 1
        return indexes_to_one_hot_tensor(y[:, :-1], input_vocab_size)

    def get_logits(self, saver, x, train_state):
        return saver.model(x, **self._forward_kwargs)

    def get_logits_and_regularization_term(self, saver, x, regularizer, train_state):
        return self.get_logits(saver, x, train_state), None

    def get_regularizer(self, args, saver):
        return torch.nn.Module()

    def get_signals(self, saver, x, train_data):
        model_type = saver.kwargs['model_type']
        more_kwargs = {}
        if model_type in ('gref', 'jm', 'ns'):
            logits, signals = saver.model(x, return_signals=True, include_first=False, **self.forward_kwargs)
            if model_type == 'ns':
                return [[x.tolist() for x in signals_t if x is not None] for signals_t in signals]
            else:
                raise NotImplementedError
        else:
            return None

    def print_example(self, saver, batch, vocab, logger):
        def get_logits(model, x):
            return self.get_logits(saver, x, None)
        print_example_outputs(saver.model, get_logits, batch, vocab, logger)

LAYER_RE = re.compile(r'^(?:(\d+)|superposition-(\d+)|nondeterministic-(\d+)-(\d+)-(\d+))$')

def parse_transformer_layers(s):
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

def get_transformer_layer_modules(
    layers,
    input_size,
    output_size,
    d_model,
    num_heads,
    feedforward_size,
    dropout
):
    # TODO Tied embeddings?
    yield get_transformer_input_unidirectional(
        input_size,
        d_model,
        dropout,
        use_padding=False
    )
    for module in get_transformer_middle_layer_modules(
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
        vocabulary_size=output_size
    )

def get_transformer_middle_layer_modules(
    layers,
    d_model,
    num_heads,
    feedforward_size,
    dropout
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
                # Don't use nested tensors, because there is no padding.
                enable_nested_tensor=False
            )
        else:
            if layer_type == 'superposition':
                stack_embedding_size, = layer_args
                attention_func = SuperpositionStackAttention(
                    d_model=d_model,
                    stack_embedding_size=stack_embedding_size
                )
            elif layer_type == 'nondeterministic':
                num_states, stack_alphabet_size, stack_embedding_size = layer_args
                attention_func = NondeterministicStackAttention(
                    d_model=d_model,
                    num_states=num_states,
                    stack_alphabet_size=stack_alphabet_size,
                    stack_embedding_size=stack_embedding_size
                )
            else:
                raise ValueError
            yield get_unidirectional_encoder_layer_with_custom_attention(
                attention_func,
                d_model=d_model,
                feedforward_size=feedforward_size,
                dropout=dropout,
                tag=layer_type
            )

def print_example_outputs(model, get_logits, batch, vocab, logger):
    x, y_target = batch
    model.eval()
    with torch.no_grad():
        # logits : B x n x V
        y_logits = get_logits(model, x)
        # y_pred : B x n
        y_pred = torch.argmax(y_logits, dim=2)
    for y_target_elem, y_pred_elem in zip(y_target.tolist(), y_pred.tolist()):
        align([
            [
                mark_color(vocab.value(p), t == p)
                for t, p
                in zip(y_target_elem, y_pred_elem)
            ],
            [vocab.value(s) for s in y_target_elem]
        ], print=logger.info)
        logger.info('')

def mark_color(s, p):
    if p:
        return green(s)
    else:
        return red(s)

def indexes_to_one_hot_tensor(indexes, vocab_size):
    result = torch.zeros(
        indexes.size() + (vocab_size,),
        device=indexes.device
    )
    if result.size(1) > 0:
        result.scatter_(2, indexes.unsqueeze(2), 1)
    return result
