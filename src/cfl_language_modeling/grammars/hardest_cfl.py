from lib.formal_models.pcfg import (
    Grammar, Rule, Nonterminal, Terminal)
from .util import mean_to_continue_prob

class HardestCFLGrammar(Grammar):

    Sp = Nonterminal("S'")
    L = Nonterminal('L')
    Lp = Nonterminal("L'")
    R = Nonterminal('R')
    Rp = Nonterminal("R'")
    U = Nonterminal('U')
    V = Nonterminal('V')
    W = Nonterminal('W')
    Q = Nonterminal('Q')
    S = Nonterminal('S')
    T = Nonterminal('T')

    lpar = Terminal(0)
    rpar = Terminal(1)
    lsq = Terminal(2)
    rsq = Terminal(3)
    comma = Terminal(4)
    semi = Terminal(5)
    dollar = Terminal(6)

    CHARS = '()[],;$'

    def __init__(self, mean_num_commas, mean_short_filler_length,
            mean_long_filler_length, semicolon_probability,
            mean_bracket_splits, mean_nesting_depth):

        assert mean_long_filler_length > 1
        assert mean_nesting_depth > 1

        Sp = self.Sp
        L = self.L
        Lp = self.Lp
        R = self.R
        Rp = self.Rp
        U = self.U
        V = self.V
        W = self.W
        Q = self.Q
        S = self.S
        T = self.T

        lpar = self.lpar
        rpar = self.rpar
        lsq = self.lsq
        rsq = self.rsq
        comma = self.comma
        semi = self.semi
        dollar = self.dollar

        c = mean_to_continue_prob(mean_num_commas)
        u = mean_to_continue_prob(mean_short_filler_length)
        v = mean_to_continue_prob(mean_long_filler_length - 1)
        q = semicolon_probability
        s = mean_to_continue_prob(mean_bracket_splits)
        t = mean_to_continue_prob(mean_nesting_depth)

        rules = []
        rules.extend([
            Rule(Sp, [R, dollar, Q, S, L, semi]),
            Rule(L, [Lp, comma, U]),
            Rule(Lp, [comma, V, Lp], c),
            Rule(Lp, [], 1 - c),
            Rule(R, [U, comma, Rp]),
            Rule(Rp, [Rp, V, comma], c),
            Rule(Rp, [], 1 - c),
            Rule(U, [W, U], u),
            Rule(U, [], 1 - u),
            Rule(V, [W, V], v),
            Rule(V, [W], 1 - v),
        ])
        for a in (lpar, rpar, lsq, rsq, dollar):
            rules.append(Rule(W, [a]))
        rules.extend([
            Rule(Q, [L, semi, R], q),
            Rule(Q, [], 1 - q),
            Rule(S, [S, Q, T], s),
            Rule(S, [T], 1 - s),
        ])
        for l, r in ((lpar, rpar), (lsq, rsq)):
            rules.extend([
                Rule(T, [l, Q, S, Q, r], t / 2),
                Rule(T, [l, Q, r], (1 - t) / 2)
            ])
        super().__init__(Sp, rules)

    @staticmethod
    def symbol_to_str(s):
        if s.is_terminal:
            return HardestCFLGrammar.int_to_str(s.value)
        else:
            return str(s)

    @staticmethod
    def int_to_str(value):
        return HardestCFLGrammar.CHARS[value]

class HardestCFLVocab:

    def value(self, i):
        if 0 <= i < self.size():
            return HardestCFLGrammar.int_to_str(i)
        else:
            raise ValueError

    def size(self):
        return 7
