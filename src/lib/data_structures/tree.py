class Tree:

    def __init__(self, value, children=None):
        if children is None:
            children = ()
        elif not isinstance(children, tuple):
            children = tuple(children)
        for child in children:
            if not isinstance(child, type(self)):
                raise TypeError('children must be trees of the same type')
        self.value = value
        self.children = children

    def __str__(self):
        if self.children:
            return '%s (%s)' % (self.value, ', '.join(str(c) for c in self.children))
        else:
            return str(self.value)

    def __repr__(self):
        if self.children:
            return 'Tree(%r, %r)' % (self.value, self.children)
        else:
            return 'Tree(%r)' % (self.value,)

    def __eq__(self, other):
        return type(self) == type(other) and self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def _key(self):
        return (self.value, self.children)

    def preorder_traversal(self):
        yield self
        for child in self.children:
            for node in child.preorder_traversal():
                yield node
