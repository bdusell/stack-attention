import collections
import itertools
import math
import random

from ..util import group_by, product
from .parse_tree import ParseTree

class LengthSampler:

    def __init__(self, grammar):
        if grammar.has_epsilon_rules or grammar.has_unary_rules:
            raise ValueError('grammar may not contain epsilon rules or unary rules')
        self.grammar = grammar
        self.rule_info = [_rule_info(r) for r in grammar.rules]
        self.indexed_rule_info = group_by(self.rule_info, lambda x: x[0].left)
        self.inside_probabilities = collections.Counter()
        self.sized_nonterminal_distributions = {}
        self.max_length = 0

    def sample(self, length, generator=random):
        self.update_inside_probabilities(length)
        return self.sample_sized_nonterminal(self.grammar.start, length, generator)

    def sample_tree(self, length, generator=random):
        self.update_inside_probabilities(length)
        return self.sample_sized_nonterminal_tree(self.grammar.start, length, generator)

    def update_inside_probabilities(self, length):
        if length > self.max_length:
            alpha = self.inside_probabilities
            for ell in range(self.max_length+1, length+1):
                for rule, size, nonterminals in self.rule_info:
                    X = rule.left
                    ell_p = ell - size
                    if not nonterminals:
                        if ell_p == 0:
                            # The rule consists of exactly ell terminals,
                            # so add the probability of the rule to
                            # alpha[X, ell].
                            alpha[X, ell] += rule.probability
                    elif ell_p >= 1:
                        # The rule has at least one nonterminal and needs
                        # to fill in exactly ell_p terminals (at least 1).
                        # Add up all the ways of generating exactly ell_p
                        # terminals from this rule's nonterminals.
                        alpha[X, ell] += rule.probability * sum(
                            product(alpha[pair] for pair in zip(nonterminals, c))
                            for c in compositions(ell_p, len(nonterminals))
                        )
            self.max_length = length

    def sample_sized_nonterminal(self, nonterminal, length, generator):
        sized_rules, cum_weights = self.get_sized_nonterminal_distribution(nonterminal, length)
        try:
            (rule, sizes), = generator.choices(sized_rules, cum_weights=cum_weights)
        except IndexError:
            raise SamplingFromEmptySet(
                f'the nonterminal {nonterminal} does not generate any strings of length {length}')
        size_it = iter(sizes)
        for X in rule.right:
            if X.is_terminal:
                yield X.value
            else:
                for a in self.sample_sized_nonterminal(X, next(size_it), generator):
                    yield a

    def sample_sized_nonterminal_tree(self, nonterminal, length, generator):
        sized_rules, cum_weights = self.get_sized_nonterminal_distribution(nonterminal, length)
        try:
            (rule, sizes), = generator.choices(sized_rules, cum_weights=cum_weights)
        except IndexError:
            raise SamplingFromEmptySet(
                f'the nonterminal {nonterminal} does not generate any strings of length {length}')
        size_it = iter(sizes)
        def get_tree(X):
            if X.is_terminal:
                return ParseTree(X)
            else:
                return self.sample_sized_nonterminal_tree(X, next(size_it), generator)
        return ParseTree(rule.left, map(get_tree, rule.right), rule)

    def is_valid_length(self, length):
        return self.is_valid_nonterminal_length(self.grammar.start, length)

    def is_valid_nonterminal_length(self, nonterminal, length):
        self.update_inside_probabilities(length)
        sized_rules, cum_weights = self.get_sized_nonterminal_distribution(nonterminal, length)
        return bool(sized_rules)

    def valid_lengths(self, length_interval):
        lo, hi = length_interval
        return [l for l in range(lo, hi+1) if self.is_valid_length(l)]

    def get_sized_nonterminal_distribution(self, nonterminal, length):
        key = nonterminal, length
        result = self.sized_nonterminal_distributions.get(key)
        if result is None:
            sized_rules = []
            weights = []
            rule_info = self.indexed_rule_info.get(nonterminal)
            if rule_info is None:
                raise SamplingFromEmptySet(f'the nonterminal {nonterminal} does not generate any strings')
            alpha = self.inside_probabilities
            for rule, size, nonterminals in rule_info:
                ell_p = length - size
                if not nonterminals:
                    if ell_p == 0:
                        sized_rules.append((rule, ()))
                        weights.append(rule.probability)
                elif ell_p >= 1:
                    for c in compositions(ell_p, len(nonterminals)):
                        weight = rule.probability * product(
                            alpha[pair]
                            for pair in zip(nonterminals, c)
                        )
                        if weight > 0:
                            sized_rules.append((rule, c))
                            weights.append(weight)
            cum_weights = list(itertools.accumulate(weights))
            result = (sized_rules, cum_weights)
            self.sized_nonterminal_distributions[key] = result
        return result

    def get_inside_probability(self, nonterminal, length):
        if nonterminal not in self.grammar.nonterminals:
            raise ValueError(f'{nonterminal} is not a nonterminal in the grammar')
        self.update_inside_probabilities(length)
        return self.inside_probabilities.get((nonterminal, length), 0.0)

    def get_inside_log_probability(self, nonterminal, length):
        return math.log(self.get_inside_probability(nonterminal, length))

    def get_length_probability(self, length):
        return self.get_inside_probability(self.grammar.start, length)

    def get_length_log_probability(self, length):
        return self.get_inside_log_probability(self.grammar.start, length)

def _rule_info(rule):
    size = 0
    nonterminals = []
    for X in rule.right:
        if X.is_terminal:
            size += 1
        else:
            nonterminals.append(X)
    return rule, size, nonterminals

def compositions(n, k):
    # See https://en.wikipedia.org/wiki/Composition_(combinatorics)
    if k == 0:
        yield ()
    elif k == 1:
        yield (n,)
    else:
        for i in range(1, n):
            for c in compositions(n - i, k - 1):
                yield (i,) + c

class SamplingFromEmptySet(Exception):
    pass
