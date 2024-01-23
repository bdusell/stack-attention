import argparse
import collections
import itertools
import math
import pathlib

import torch
import torch_semiring_einsum

from lib.semiring import log_viterbi
from stack_rnn_models.nondeterministic_stack import (
    logits_to_actions,
    get_nondeterministic_stack,
    ViterbiDecoder,
    PushOperation,
    ReplaceOperation,
    PopOperation
)
from cfl_language_modeling.model_util import CFLModelInterface
from cfl_language_modeling.task_util import add_task_arguments, parse_task
from utils.plot_util import add_plot_arguments, run_plot
from cfl_language_modeling.plot_stack_attention_heatmap import (
    convert_tokens_to_input_tensor,
    format_symbol_label
)
from lib.pretty_table import align

def read_input_tokens(path):
    with path.open() as fin:
        return fin.readline().split()

def stack_symbol_to_str(y, use_safe_latex):
    if y == 0:
        return '\\bot'
    else:
        func_name = 'mathtt' if use_safe_latex else 'sym'
        return f'\\{func_name}{{{y-1}}}'

def actions_to_viterbi_decoder(actions):
    with torch.no_grad():
        tensor = actions[0][0]
        batch_size, num_states, stack_alphabet_size, *_ = tensor.size()
        sequence_length = len(actions)
        # Compute the gamma and alpha tensor for every timestep in the
        # Viterbi semiring.
        stack = get_nondeterministic_stack(
            batch_size=batch_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            sequence_length=sequence_length,
            include_states_in_reading=True,
            normalize_reading=True,
            block_size=torch_semiring_einsum.AUTOMATIC_BLOCK_SIZE,
            dtype=tensor.dtype,
            device=tensor.device,
            semiring=log_viterbi
        )
        # The first `result` returned from `update` corresponds to timestep
        # j = 1, so these lists include results starting just before
        # timestep j = 2.
        alpha_columns = []
        gamma_j_nodes = []
        alpha_j_nodes = []
        for push, repl, pop in actions:
            result = stack.update(
                log_viterbi.primitive(push),
                log_viterbi.primitive(repl),
                log_viterbi.primitive(pop)
            )
            # Save the nodes for the columns of gamma and alpha in lists.
            # This makes decoding simpler.
            alpha_columns.append(result.alpha_j)
            gamma_j_nodes.append(result.gamma_j[1])
            alpha_j_nodes.append(result.alpha_j[1])
    return ViterbiDecoder(alpha_columns, gamma_j_nodes, alpha_j_nodes)

def get_distinct_viterbi_paths(decoder, n):
    # This is not an efficient algorithm. :(
    distinct_paths = collections.OrderedDict()
    for i in reversed(range(1, n+1)):
        (path,), scores = decoder.decode_timestep(i)
        path = tuple(path)
        is_distinct = True
        for existing_path in distinct_paths.keys():
            if is_prefix_of(path, existing_path):
                distinct_paths[existing_path] += 1
                is_distinct = False
        if is_distinct:
            distinct_paths[path] = 1
    return distinct_paths

def is_prefix_of(list1, list2):
    return len(list1) <= len(list2) and list1 == list2[:len(list1)]

def format_stack_symbol(s):
    if s == 0:
        return 'bot'
    else:
        return str(s-1)

def format_operation(op):
    if isinstance(op, PushOperation):
        return f'push {op.state_to}, {format_stack_symbol(op.symbol)}'
    elif isinstance(op, ReplaceOperation):
        return f'repl {op.state_to}, {format_stack_symbol(op.symbol)}'
    elif isinstance(op, PopOperation):
        return f'pop {op.state_to}'
    else:
        raise TypeError

def add_stack_depths(ops):
    depth = 0
    for op in ops:
        yield op, depth
        if isinstance(op, PushOperation):
            depth += 1
        elif isinstance(op, ReplaceOperation):
            pass
        elif isinstance(op, PopOperation):
            depth -= 1
        else:
            raise TypeError
        if depth < 0:
            raise ValueError

def pad_lists(rows, pad_value=''):
    max_len = max(map(len, rows))
    for row in rows:
        while len(row) < max_len:
            row.append('')

def transpose_jagged_lists(rows):
    pad_lists(rows)
    return zip(*rows)

def run_main(
    model_interface_class,
    add_vocab_arguments,
    load_vocab,
    construct_saver,
    get_layers,
    convert_tokens_to_input_tensor,
    format_symbol_label
):

    model_interface = model_interface_class(use_load=True, use_init=False, use_output=False)

    parser = argparse.ArgumentParser()
    model_interface.add_arguments(parser)
    add_vocab_arguments(parser)
    parser.add_argument('--input-string', type=pathlib.Path, required=True)
    args = parser.parse_args()

    device = model_interface.get_device(args)
    vocabs = load_vocab(parser, args)

    saver = construct_saver(model_interface, args, vocabs)
    model = saver.model

    transformer_layers = get_layers(saver)
    if not transformer_layers:
        raise ValueError
    stack_attention_layers = [name for name, _ in transformer_layers if name != 'transformer']
    if stack_attention_layers not in (['superposition'], ['nondeterministic']):
        raise ValueError
    stack_attention_type, = stack_attention_layers

    input_string_strs = read_input_tokens(args.input_string)
    input_tensor = convert_tokens_to_input_tensor(
        model_interface,
        vocabs,
        device,
        input_string_strs
    )

    with torch.no_grad():
        model.eval()
        result = model(
            input_tensor,
            tag_kwargs=dict(
                superposition=dict(
                    return_actions=True
                ),
                nondeterministic=dict(
                    block_size=torch_semiring_einsum.AUTOMATIC_BLOCK_SIZE,
                    return_actions=True
                )
            )
        )
        actions = result.extra_outputs[0][:, 0]
        if stack_attention_type == 'superposition':
            num_states = stack_alphabet_size = 1
            # actions : sequence_length x 3
            # Reorder from push, pop, no-op to push, no-op, pop
            actions = torch.log(actions[:, (0, 2, 1)])
        elif stack_attention_type == 'nondeterministic':
            for name, layer_args in transformer_layers:
                if name == 'nondeterministic':
                    break
            num_states, stack_alphabet_size, _ = layer_args
        else:
            raise ValueError

    push, repl, pop = logits_to_actions(actions, num_states, stack_alphabet_size, False)
    actions = [
        tuple(op[None] for op in ops)
        for ops in zip(push, repl, pop)
    ]
    viterbi_decoder = actions_to_viterbi_decoder(actions)
    paths = get_distinct_viterbi_paths(viterbi_decoder, input_tensor.size(1))
    paths_and_counts = sorted(paths.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
    rows = [[s] for s in ['BOS', *input_string_strs]]
    for path, count in paths_and_counts:
        for row, op_and_depth in itertools.zip_longest(rows, add_stack_depths(path)):
            if op_and_depth is not None:
                op, depth = op_and_depth
                row.extend('' for i in range(depth))
                row.append(format_operation(op))
        pad_lists(rows)
    horizontal = False
    if horizontal:
        rows = zip(*rows)
    align(rows, max_table_width=math.inf)

def main():
    run_main(
        model_interface_class=CFLModelInterface,
        add_vocab_arguments=add_task_arguments,
        load_vocab=parse_task,
        construct_saver=lambda model_interface, args, task: model_interface.construct_saver(
            args,
            input_size=task.input_vocab.size(),
            output_size=task.output_vocab.size()
        ),
        get_layers=lambda saver: saver.kwargs['transformer_layers'],
        convert_tokens_to_input_tensor=convert_tokens_to_input_tensor,
        format_symbol_label=format_symbol_label
    )

if __name__ == '__main__':
    main()
