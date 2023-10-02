import math

import torch

def recursively_stack(tensors, dims=None, device=None):
    if isinstance(tensors, list):
        if tensors:
            d = dims[1:] if dims is not None else None
            return torch.stack([recursively_stack(t, d) for t in tensors])
        else:
            if dims is None:
                raise ValueError
            size = list(dims)
            size[0] = 0
            return torch.empty(size, device=device)
    else:
        if not isinstance(tensors, torch.Tensor):
            tensors = torch.tensor(float(tensors))
        return tensors

def logsumexp(scalars, device=None):
    if scalars:
        return torch.logsumexp(torch.stack(scalars), dim=0)
    else:
        return torch.tensor(0, device=device)

def logaddexp(a, b):
    c = torch.max(a, b)
    return c + torch.log(torch.exp(a - c) + torch.exp(b - c))
