from lib.formal_models.pcfg import (
    Grammar, Rule, Nonterminal, Terminal)
from .util import mean_to_continue_prob

BRACKET_STRINGS = '()[]{}<>'

class DyckGrammar(Grammar):

    S = Nonterminal('S')
    T = Nonterminal('T')

    def __init__(self, bracket_types, mean_bracket_splits, mean_nesting_depth):
        assert mean_nesting_depth > 1
        S = self.S
        T = self.T
        split_prob = mean_to_continue_prob(mean_bracket_splits)
        nest_prob = mean_to_continue_prob(mean_nesting_depth - 1)
        rules = []
        rules.extend([
            Rule(S, [S, T], split_prob),
            Rule(S, [T], 1 - split_prob)
        ])
        for i in range(bracket_types):
            l = Terminal(2 * i)
            r = Terminal(2 * i + 1)
            rules.append(Rule(T, (l, S, r), nest_prob))
            rules.append(Rule(T, (l, r), 1 - nest_prob))
        super().__init__(S, rules)

class DyckVocab:

    def __init__(self, bracket_types):
        super().__init__()
        self.bracket_types = bracket_types

    def value(self, i):
        if 0 <= i < self.size():
            div, index = divmod(i, len(BRACKET_STRINGS))
            return BRACKET_STRINGS[index] * (div+1)
        else:
            raise ValueError

    def size(self):
        return 2 * self.bracket_types
