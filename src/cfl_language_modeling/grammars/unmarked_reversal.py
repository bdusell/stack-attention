from lib.formal_models.pcfg import (
    Grammar, Rule, Nonterminal, Terminal)
from .util import mean_to_continue_prob

class UnmarkedReversalGrammar(Grammar):

    S = Nonterminal('S')

    def __init__(self, symbol_types, mean_length):
        S = self.S
        nesting_prob = mean_to_continue_prob(mean_length)
        nest_a_prob = nesting_prob / symbol_types
        term_prob = 1 - nesting_prob
        rules = []
        for i in range(symbol_types):
            a = Terminal(i)
            rules.append(Rule(S, (a, S, a), nest_a_prob))
        rules.append(Rule(S, (), term_prob))
        super().__init__(S, rules)
        self.symbol_types = symbol_types
        self.mean_length = mean_length
        self.nesting_probability = nesting_prob
        self.terminating_probability = term_prob

class UnmarkedReversalVocab:

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
