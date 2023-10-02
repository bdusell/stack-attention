def remove_epsilon_rules(grammar):
    _remove_rules_of_type(grammar, lambda r: r.is_epsilon)

def remove_unary_rules(grammar):
    _remove_rules_of_type(grammar, lambda r: r.is_unary)

def _remove_rules_of_type(grammar, predicate):
    rules = grammar.rules
    while True:
        for i, rule in enumerate(rules):
            if predicate(rule):
                break
        else:
            break
        rules = list(_remove_rule(rules, i, rule))
    grammar.rules = rules

def _remove_rule(rules, i, removed_rule):
    Rule = type(removed_rule)
    for j, rule in enumerate(rules):
        if j != i:
            for right, prob in _replacements(rule.right, removed_rule):
                new_prob = rule.probability * prob
                if rule.left == removed_rule.left:
                    new_prob /= 1.0 - removed_rule.probability
                yield Rule(rule.left, right, new_prob)

def _replacements(sequence, rule):
    p = rule.probability
    q = 1.0 - p
    indexes = [i for i, X in enumerate(sequence) if X == rule.left]
    for mask in _power_set_masks(len(indexes)):
        result = []
        prob = 1.0
        i = 0
        for j, m in zip(indexes, mask):
            result.extend(sequence[i:j])
            if m:
                # Replace the nonterminal
                result.extend(rule.right)
                prob *= p
            else:
                # Keep the nonterminal
                result.append(sequence[j])
                prob *= q
            i = j + 1
        result.extend(sequence[i:])
        yield result, prob

def _power_set_masks(n):
    for i in range(1 << n):
        yield tuple(bool(i & (1 << j)) for j in range(n))
