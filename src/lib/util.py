import collections

def group_by(iterable, key):
    result = collections.defaultdict(list)
    for value in iterable:
        result[key(value)].append(value)
    return result

def product(iterable):
    result = 1
    for value in iterable:
        result *= value
    return result

def dfs(root, children):
    visited = set()
    def visit(node):
        yield node
        visited.add(node)
        for child in children(node):
            if child not in visited:
                for node2 in visit(child):
                    yield node2
    return visit(root)

def to_list(values):
    if not isinstance(values, (list, tuple)):
        values = list(values)
    return values
