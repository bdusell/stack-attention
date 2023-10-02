import math

from .cfg import Grammar, Rule, Terminal, Nonterminal
from ..util import group_by

class Rule(Rule):

    def __init__(self, left, right, probability=1.0):
        super().__init__(left, right)
        if probability < 0:
            raise ValueError('probability cannot be negative')
        self.probability = probability

    @property
    def log_probability(self):
        return math.log(self.probability)

    def _get_kwargs(self):
        return dict(
            left=self.left,
            right=self.right,
            probability=self.probability
        )

    def __str__(self):
        return '{} [{}]'.format(super().__str__(), self.probability)

    def __repr__(self):
        return 'Rule({!r}, {!r}, {!r})'.format(self.left, self.right, self.probability)

class Grammar(Grammar):

    rule_type = Rule

    def __init__(self, start, rules, normalize=True):
        super().__init__(start, rules)
        if normalize:
            self.normalize_rule_probabilities()

    def normalize_rule_probabilities(self):
        grouped_rules = group_by(self.rules, key=lambda r: r.left)
        for A, rules in grouped_rules.items():
            denom = sum(r.probability for r in rules)
            for rule in rules:
                rule.probability /= denom
