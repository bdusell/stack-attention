from lib.formal_models.pcfg import (
    Grammar, Rule, Nonterminal, Terminal)
from .util import mean_to_continue_prob

class PaddedReversalGrammar(Grammar):

    S = Nonterminal('S')

    def __init__(self, symbol_types, mean_content_length, mean_padding_length):
        S = self.S
        nesting_prob = mean_to_continue_prob(mean_content_length)
        nesting_prob_a = nesting_prob / symbol_types
        stop_nesting_prob = 1 - nesting_prob
        stop_nesting_prob_a = stop_nesting_prob / symbol_types
        gen_pad_prob = mean_to_continue_prob(mean_padding_length)
        stop_pad_prob = 1 - gen_pad_prob
        rules = []
        for i in range(symbol_types):
            a = Terminal(i)
            Ta = Nonterminal(f'T{i}')
            rules.append(Rule(S, (a, S, a), nesting_prob_a))
            rules.append(Rule(S, (Ta,), stop_nesting_prob_a))
            rules.append(Rule(Ta, (a, Ta), gen_pad_prob))
            rules.append(Rule(Ta, (), stop_pad_prob))
        super().__init__(S, rules)
        self.symbol_types = symbol_types
        self.mean_content_length = mean_content_length
        self.mean_padding_length = mean_padding_length
        self.nesting_probability = nesting_prob
        self.stop_nesting_probability = stop_nesting_prob
        self.generate_padding_probability = gen_pad_prob
        self.stop_padding_probabililty = stop_pad_prob

class PaddedReversalVocab:

    def __init__(self, symbol_types):
        super().__init__()
        self.symbol_types = symbol_types

    def value(self, i):
        if 0 <= i < self.symbol_types:
            return str(i)
        else:
            raise ValueError

    def size(self):
        return self.symbol_types
