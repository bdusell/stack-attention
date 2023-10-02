import math

import torch

from lib.util import to_list

class MemoryLookup:

    def __init__(self, values):
        super().__init__()
        self.table = construct_table(values)

    def lookup(self, n, m):
        try:
            return self.table[n, m].item()
        except IndexError:
            return math.inf

def construct_table(values):
    # This is a simple dynamic programming algorithm that computes, for every
    # source length n and target length m, the smallest value of any (n', m')
    # such that n <= n' and m <= m'.
    # NOTE: This assumes that the actual memory usage increases monotonically
    # with input size, which in practice should always hold true.
    values = to_list(values)
    max_n = max(n for (n, m, v) in values)
    max_m = max(m for (n, m, v) in values)
    table = torch.full((max_n+1, max_m+1), math.inf, dtype=torch.float)
    for n, m, v in values:
        table[n, m] = v
    for n in reversed(range(max_n+1)):
        for m in reversed(range(max_m+1)):
            args = [table[n, m]]
            if n+1 <= max_n:
                args.append(table[n+1, m])
            if m+1 <= max_m:
                args.append(table[n, m+1])
            table[n, m] = torch.min(torch.stack(args))
    return table
