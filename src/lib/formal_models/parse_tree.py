from ..data_structures.tree import Tree
from ..util import product

class ParseTree(Tree):

    def __init__(self, symbol, children=None, rule=None):
        super().__init__(symbol, children)
        if symbol.is_terminal:
            if self.children:
                raise ValueError('a terminal cannot have children')
            if rule is not None:
                raise ValueError('a terminal cannot have a rule')
        if rule is not None:
            if not (
                rule.left == symbol and
                all(x == y.symbol for (x, y) in zip(rule.right, self.children))
            ):
                raise ValueError('rule does not match tree structure')
        self.rule = rule

    @property
    def symbol(self):
        return self.value

    @property
    def terminals(self):
        if self.symbol.is_terminal:
            yield self.symbol
        else:
            for child in self.children:
                for s in child.terminals:
                    yield s

    @property
    def probability(self):
        if self.rule is None:
            return 1.0
        else:
            return (
                self.rule.probability *
                product(c.probability for c in self.children)
            )

    @property
    def log_probability(self):
        if self.rule is None:
            return 0.0
        else:
            return (
                self.rule.log_probability +
                sum(c.log_probability for c in self.children)
            )

def left_parse_to_tree(grammar, parse):
    it = iter(parse)
    def process_symbol(symbol):
        if symbol.is_terminal:
            return ParseTree(symbol)
        else:
            try:
                next_rule = next(it)
            except StopIteration:
                raise ValueError('parse is too short')
            if next_rule.left != symbol:
                raise ValueError('next rule in parse does not expand the next symbol')
            return process_rule(next_rule)
    def process_rule(rule):
        return ParseTree(rule.left, (process_symbol(s) for s in rule.right), rule)
    try:
        first_rule = next(it)
    except StopIteration:
        raise ValueError('parse cannot be empty')
    if first_rule.left != grammar.start:
        raise ValueError('first rule in parse does not expand the start symbol')
    result = process_rule(first_rule)
    try:
        next(it)
    except StopIteration:
        return result
    else:
        raise ValueError('parse is too long')
