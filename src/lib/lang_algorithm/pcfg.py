import collections
import math

from ..util import group_by

class NoParses(ValueError):
    pass

def compute_parse_grammar_log_probability(parses):
    rules_by_left = group_by(parses.rules, key=lambda r: r.left)
    visiting = set()
    cache = collections.defaultdict(float)
    def recurse(symbol):
        if symbol.is_terminal:
            # This "terminal" is a PCFG rule. Its probability is the
            # probability of the rule.
            return symbol.value.log_probability
        else:
            if symbol in visiting:
                # TODO This can probably be handled in the general case.
                raise ValueError('parse grammar has recursion')
            result = cache.get(symbol)
            if result is None:
                rules = rules_by_left.get(symbol)
                if rules is None:
                    raise NoParses('a nonterminal produces no parses')
                else:
                    visiting.add(symbol)
                    result = logspace_sum(
                        logspace_product(recurse(X) for X in rule.right)
                        for rule in rules
                    )
                    visiting.remove(symbol)
                cache[symbol] = result
            return result
    return recurse(parses.start)

def string_log_probability(parser, string):
    parses = parser.to_parse_grammar(string)
    try:
        ll = compute_parse_grammar_log_probability(parses)
    except NoParses:
        return -math.inf
    else:
        return ll

logspace_product = sum

def logspace_sum(values):
    return math.log(sum(math.exp(x) for x in values))
