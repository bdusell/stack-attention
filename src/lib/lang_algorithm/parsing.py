from ..util import group_by
from ..formal_models.parse_tree import left_parse_to_tree
from .pdt import Transducer, Transition
from .lang import TransducerIndex, simulate_transducer, initial_step

START_STATE = 0
LOOP_STATE = 1

def grammar_to_transducer(grammar):
    return Transducer(
        start_state=START_STATE,
        transitions=_grammar_to_transducer_transitions(grammar),
        accept_states=[LOOP_STATE]
    )

def _grammar_to_transducer_transitions(grammar):
    yield Transition(
        state_from=START_STATE,
        state_to=LOOP_STATE,
        pushed_symbol=grammar.start
    )
    for symbol in grammar.terminals:
        yield Transition(
            state_from=LOOP_STATE,
            input_symbol=symbol,
            popped_symbol=symbol,
            state_to=LOOP_STATE
        )
    state = 2
    for rule in grammar.rules:
        yield Transition(
            state_from=LOOP_STATE,
            popped_symbol=rule.left,
            state_to=state,
            output_string=[rule]
        )
        for symbol in reversed(rule.right):
            yield Transition(
                state_from=state,
                state_to=state + 1,
                pushed_symbol=symbol
            )
            state += 1
        yield Transition(
            state_from=state,
            state_to=LOOP_STATE
        )
        state += 1

def enumerate_strings_in_grammar(grammar):
    rule_map = group_by(grammar.rules, key=lambda r: r.left)
    visiting = set()
    def _enumerate_strings_from_symbol(symbol):
        if symbol.is_terminal:
            yield [symbol.value]
        else:
            if symbol in visiting:
                raise ValueError(
                    'the parse grammar has recursion; there are an infinite '
                    'number of valid parses')
            visiting.add(symbol)
            rules = rule_map.get(symbol)
            if rules is not None:
                for rule in rules:
                    choices = [
                        list(_enumerate_strings_from_symbol(s))
                        for s in rule.right
                    ]
                    for choice in enumerate_choices(choices):
                        yield sum(choice, [])
            visiting.remove(symbol)
    return _enumerate_strings_from_symbol(grammar.start)

def enumerate_choices(choices):
    def _enumerate(i):
        if i == len(choices):
            yield []
        else:
            for choice1 in choices[i]:
                for choice2 in _enumerate(i + 1):
                    yield [choice1] + choice2
    return _enumerate(0)

class Parser:

    def __init__(self, grammar):
        self.grammar = grammar
        self.transducer_index = TransducerIndex(grammar_to_transducer(grammar))

    def parse(self, string):
        parse_grammar = self.to_parse_grammar(string)
        for parse in enumerate_strings_in_grammar(parse_grammar):
            yield left_parse_to_tree(self.grammar, parse)

    def to_parse_grammar(self, string):
        return simulate_transducer(self.transducer_index, string)

    def initial_step(self):
        return initial_step(self.transducer_index)
