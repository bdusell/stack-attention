def parse_interval(s):
    lo, hi = s.split(':', 1)
    return int(lo), int(hi)

def get_kwargs(parser, args, names):
    result = {}
    for name in names:
        value = getattr(args, name)
        if value is None:
            parser.error('the arguments {} are required'.format(', '.join(names)))
        result[name] = value
    return result
