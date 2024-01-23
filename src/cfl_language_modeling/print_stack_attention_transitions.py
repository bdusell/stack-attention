import itertools

import torch

from cfl_language_modeling.analyze_stack_attention_util import AnalyzeStackAttention
from cfl_language_modeling.cfl_stack_attention_util import CFLAdapter

def generate_actions(num_states, stack_alphabet_size):
    for q in range(num_states):
        for x in range(stack_alphabet_size):
            for r in range(num_states):
                for y in range(stack_alphabet_size):
                    yield 'push', (q, x, r, y)
            for r in range(num_states):
                for y in range(stack_alphabet_size):
                    yield 'replace', (q, x, r, y)
            for r in range(num_states):
                yield 'pop', (q, x, r)

def generate_transitions(input_token_strs, actions, num_states, stack_alphabet_size, threshold):
    input_token_indexes = {}
    for i, s in enumerate(itertools.chain(['BOS'], input_token_strs)):
        if s not in input_token_indexes:
            input_token_indexes[s] = []
        input_token_indexes[s].append(i)
    input_token_actions = [
        (s, torch.mean(actions[indexes], dim=0))
        for s, indexes in input_token_indexes.items()
    ]
    min_weight = torch.min(actions).item()
    max_weight = torch.max(actions).item()
    threshold_weight = min_weight + threshold * (max_weight - min_weight)
    for input_token_type, s_actions in input_token_actions:
        #min_weight = torch.min(s_actions).item()
        #max_weight = torch.max(s_actions).item()
        #threshold_weight = min_weight + threshold * (max_weight - min_weight)
        for (action_type, action_args), weight \
                in zip(generate_actions(num_states, stack_alphabet_size), s_actions.tolist()):
            if weight >= threshold_weight:
                yield (input_token_type, action_type, action_args, weight)

def format_state(i):
    return f'q_{i}'

def format_stack_symbol(i):
    if i == 0:
        return '\\bot'
    else:
        return f'\\sym{{{i-1}}}'

def format_input_symbol(s):
    if s == 'BOS':
        return '\\bos{}'
    else:
        s = s.replace('#', '\\#').replace('$', '\\$')
        return f'\\sym{{{s}}}'

class PrintTransitions(CFLAdapter):

    def add_arguments(self, parser):
        pass

    def run(self, args, stack_attention_type, stack_attention_args, input_token_strs, actions):
        if stack_attention_type != 'nondeterministic':
            raise ValueError
        num_states, stack_alphabet_size, _ = stack_attention_args
        for input_token_type, action_type, action_args, weight in \
                generate_transitions(input_token_strs, actions, num_states, stack_alphabet_size, 0.95):
            q = format_state(action_args[0])
            x = format_stack_symbol(action_args[1])
            a = format_input_symbol(input_token_type)
            r = format_state(action_args[2])
            if action_type == 'pop':
                v = '\\emptystring'
            else:
                y = format_stack_symbol(action_args[3])
                if action_type == 'replace':
                    v = y
                else:
                    v = f'{x}{y}'
            print(f'{q}, {a}, {x} \\rightarrow, {r}, {v}')

class Program(PrintTransitions, CFLAdapter):
    pass

Program().main()
