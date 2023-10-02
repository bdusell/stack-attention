class Symbol:

    def __init__(self, value):
        super().__init__()
        self.value = value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f'Symbol({self.value!r})'

    def __eq__(self, other):
        return type(self) == type(other) and self.value == other.value

    def __hash__(self):
        return hash((type(self), self.value))

    def __lt__(self, other):
        if not isinstance(other, Symbol):
            raise TypeError
        return self._order_key() < other._order_key()

    def _order_key(self):
        return (self.is_terminal, self.value)

    @property
    def is_terminal(self):
        raise NotImplementedError

    @property
    def is_nonterminal(self):
        raise NotImplementedError

class Terminal(Symbol):

    def __repr__(self):
        return f'Terminal({self.value!r})'

    @property
    def is_terminal(self):
        return True

    @property
    def is_nonterminal(self):
        return False

class Nonterminal(Symbol):

    def __repr__(self):
        return f'Nonterminal({self.value!r})'

    @property
    def is_terminal(self):
        return False

    @property
    def is_nonterminal(self):
        return True

class Rule:

    def __init__(self, left, right):
        if not isinstance(left, Nonterminal):
            raise TypeError('left side must be a nonterminal')
        if not isinstance(right, tuple):
            right = tuple(right)
        for x in right:
            if not isinstance(x, Symbol):
                raise TypeError('right side must be a sequence of symbols')
        self.left = left
        self.right = right

    @property
    def is_epsilon(self):
        return len(self.right) == 0

    @property
    def is_unary(self):
        return len(self.right) == 1 and self.right[0].is_nonterminal

    def replaced(self, **kwargs):
        _kwargs = self._get_kwargs()
        _kwargs.update(kwargs)
        return type(self)(**_kwargs)

    def _get_kwargs(self):
        return dict(left=self.left, right=self.right)

    def _key(self):
        return (self.left, self.right)

    def __hash__(self):
        return hash(self._key())

    def __eq__(self, other):
        return type(self) == type(other) and self._key() == other._key()

    def __str__(self):
        return '%s -> %s' % (
            self.left,
            ' '.join(map(str, self.right))
        )

    def __repr__(self):
        return 'Rule(%r, %r)' % (self.left, self.right)

class Grammar:

    rule_type = Rule

    def __init__(self, start, rules):
        if not isinstance(start, Nonterminal):
            raise TypeError('start symbol must be a nonterminal')
        if not isinstance(rules, list):
            rules = list(rules)
        for rule in rules:
            if not isinstance(rule, self.rule_type):
                raise TypeError('rules must be instances of %s' % self.rule_type)
        terminals = set()
        nonterminals = set()
        for rule in rules:
            nonterminals.add(rule.left)
            for X in rule.right:
                (terminals if X.is_terminal else nonterminals).add(X)
        self.start = start
        self.rules = rules
        self.terminals = terminals
        self.nonterminals = nonterminals

    @property
    def has_epsilon_rules(self):
        return any(r.is_epsilon for r in self.rules)

    @property
    def has_unary_rules(self):
        return any(r.is_unary for r in self.rules)

    def __str__(self):
        return '\n'.join(map(str, self.rules))

    def __repr__(self):
        return 'Grammar(%r, %r)' % (self.start, self.rules)
