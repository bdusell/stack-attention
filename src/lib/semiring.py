import math

import torch_semiring_einsum
import torch

class Semiring:

    def __init__(self, multiply, add, set, einsum):
        self.multiply_op = multiply
        self.add_op = add
        self.set = set
        self._einsum = einsum

    def multiply(self, a, b):
        return self.multiply_op.binary(a, b)

    def multiply_in_place(self, a, b):
        self.multiply_op.in_place(a, b)

    def product(self, x, dim):
        return self.multiply_op.nary(x, dim)

    def add(self, a, b):
        return self.add_op.binary(a, b)

    def add_in_place(self, a, b):
        self.add_op.in_place(a, b)

    def sum(self, x, dim):
        return self.add_op.nary(x, dim)

    def sum_multi(self, x, dims):
        return self.add_op.nary_multi(x, dims)

    def on_tensor(self, x, f):
        return self.set.on_tensor(x, f)

    def get_tensor(self, x):
        return self.set.get_tensor(x)

    def primitive(self, x):
        return self.set.primitive(x)

    def zeros(self, size, device=None, dtype=None, like=None):
        if like is not None:
            like_tensor = self.get_tensor(like)
            device = like_tensor.device
            dtype = like_tensor.dtype
        return self.set.zeros(size, device, dtype)

    def ones(self, size, device=None, dtype=None, like=None):
        if like is not None:
            like_tensor = self.get_tensor(like)
            device = like_tensor.device
            dtype = like_tensor.dtype
        return self.set.ones(size, device, dtype)

    def combine(self, args, f):
        return self.set.combine(args, f)

    def split(self, arg, f):
        return self.set.split(arg, f)

    def mark(self, x, f):
        return self.set.mark(x, f)

    def einsum(self, equation, *args, **kwargs):
        return self.combine(
            args,
            lambda args: self._einsum(equation, *args, **kwargs))

    @property
    def zero(self):
        return self.set.zero

    @property
    def one(self):
        return self.set.one

class MultiplyOperation:

    def binary(self, a, b):
        raise NotImplementedError

    def in_place(self, a, b):
        raise NotImplementedError

    def nary(self, x, dim):
        raise NotImplementedError

class AddOperation:

    def binary(self, a, b):
        raise NotImplementedError

    def in_place(self, a, b):
        raise NotImplementedError

    def nary(self, x, dim):
        raise NotImplementedError

    def nary_multi(self, x, dims):
        for i in range(dims):
            x = self.nary(x, 0)
        return x

class Set:

    def __init__(self, zero, one):
        self.zero = zero
        self.one = one

    def on_tensor(self, x, f):
        raise NotImplementedError

    def get_tensor(self, x):
        raise NotImplementedError

    def primitive(self, x):
        raise NotImplementedError

    def zeros(self, size, device, dtype):
        return self.primitive(torch.full(
            size, self.zero, device=device, dtype=dtype))

    def ones(self, size, device, dtype):
        return self.primitive(torch.full(
            size, self.one, device=device, dtype=dtype))

    def combine(self, args, f):
        raise NotImplementedError

    def split(self, arg, f):
        raise NotImplementedError

    def mark(self, x, f):
        raise NotImplementedError

class RealMultiply(MultiplyOperation):

    def binary(self, a, b):
        return a * b

    def in_place(self, a, b):
        a.mul_(b)

    def nary(self, x, dim):
        return torch.prod(x, dim)

class RealAdd(AddOperation):

    def binary(self, a, b):
        return a + b

    def in_place(self, a, b):
        a.add_(b)

    def nary(self, x, dim):
        return torch.sum(x, dim)

LogMultiply = RealAdd

class LogAdd(AddOperation):

    def binary(self, a, b):
        c = torch.max(a, b)
        # Clipping to `min_float` fixes an edge case where a and b are both
        # -inf (the problem is that (-inf - -inf) produces nan).
        min_float = c.new_tensor(torch.finfo(c.dtype).min)
        c = torch.max(c, min_float)
        return c + torch.log(torch.exp(a - c) + torch.exp(b - c))

    def nary(self, x, dim):
        return torch.logsumexp(x, dim)

class Node:

    def __init__(self, backpointers, children):
        self.backpointers = backpointers
        self.children = children

class SplitNode:

    def __init__(self, node, index):
        self.node = node
        self.index = index

class MarkedNode:

    def __init__(self, node, mark):
        self.node = node
        self.mark = mark

class LogViterbiMultiply(LogMultiply):

    def binary(self, a, b):
        a_value, a_node = a
        b_value, b_node = b
        value = super().binary(a_value, b_value)
        node = Node(None, (a_node, b_node))
        return value, node

    def nary(self, x, dim):
        raise NotImplementedError

def viterbi_primitive(tensor):
    return tensor, Node(None, ())

class ViterbiAdd(AddOperation):

    def binary(self, a, b):
        a_value, a_node = a
        b_value, b_node = b
        argmax = torch.lt(a_value, b_value)
        max_values = torch.max(a_value, b_value)
        node = Node(argmax, (a_node, b_node))
        return max_values, node

    def nary(self, x, dim):
        x_value, x_node = x
        max_values, argmax = torch.max(x_value, dim=dim)
        node = Node(argmax, (x_node,))
        return max_values, node

    def nary_multi(self, x, dims):
        # x : K1 x K2 x ... Kn x N1 x N2 x ... x Nm
        # return : (N1 x N2 x ... x Nm, N1 x N2 x ... x Nm x n)
        max_values, x_backtrace = x
        argmax_indexes = []
        for i in range(dims):
            max_values, argmax_i = torch.max(max_values, dim=0)
            # max_values : Ki x ... x Kn x N1 x ... x Nm
            # argmax_i   : Ki x ... x Kn x N1 x ... x Nm
            # argmax_indexes : i x [ Ki-1 x ... x Kn x N1 x ... x Nm ]
            argmax_indexes = [lookup_indexes(a, argmax_i) for a in argmax_indexes]
            argmax_indexes.append(argmax_i)
        argmax = torch.stack(argmax_indexes, dim=-1)
        return max_values, Node(argmax, (x_backtrace,))

def lookup_indexes(prev, curr):
    # prev : K x N1 x ... x Nm
    # curr : N1 x ... x Nm, with int values in [0, K)
    # return : N1 x ... x Nm
    # return[n1, n2, ..., nm] = prev[curr[n1, n2, ..., nm], n1, n2, ..., nm]
    return torch.gather(prev, 0, curr.unsqueeze(0)).squeeze(0)

class ScalarSet(Set):

    def on_tensor(self, x, f):
        return f(x)

    def get_tensor(self, x):
        return x

    def primitive(self, x):
        return x

    def combine(self, args, f):
        return f(args)

    def split(self, arg, f):
        return f(arg)

    def mark(self, x, f):
        return x

class ViterbiSet(Set):

    def on_tensor(self, x, f):
        return f(x[0]), x[1]

    def get_tensor(self, x):
        return x[0]

    def primitive(self, x):
        return x, Node(None, ())

    def combine(self, args, f):
        tensors = []
        nodes = []
        for tensor, node in args:
            tensors.append(tensor)
            nodes.append(node)
        return f(tensors), Node(None, nodes)

    def split(self, arg, f):
        tensor, node = arg
        return [
            (x, SplitNode(node, i))
            for i, x in enumerate(f(tensor))
        ]

    def mark(self, x, f):
        return x[0], MarkedNode(x[1], f(x[1]))

class EinsumFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, forward, backward, equation, *args):
        if forward is None:
            raise NotImplementedError
        ctx.forward = forward
        ctx.backward = backward
        ctx.equation = equation
        if backward is not None:
            ctx.save_for_backward(*args)
        return forward(equation, *args)

    @staticmethod
    def backward(ctx, grad):
        if ctx.backward is None:
            raise NotImplementedError
        args = ctx.saved_tensors
        needs_grad = ctx.needs_input_grad[3:]
        input_grads = ctx.backward(ctx.equation, args, needs_grad, grad)
        return (None, None, None) + tuple(input_grads)

class ViterbiSemiring(Semiring):

    def einsum(self, equation, *args, **kwargs):
        (max_values, argmax), node = super().einsum(equation, *args, **kwargs)
        node.backpointers = argmax
        return max_values, node

real = Semiring(
    RealMultiply(),
    RealAdd(),
    ScalarSet(zero=0.0, one=1.0),
    torch_semiring_einsum.einsum
)
log = Semiring(
    LogMultiply(),
    LogAdd(),
    ScalarSet(zero=-math.inf, one=0.0),
    torch_semiring_einsum.log_einsum
)
log_viterbi = ViterbiSemiring(
    LogViterbiMultiply(),
    ViterbiAdd(),
    ViterbiSet(zero=-math.inf, one=0.0),
    torch_semiring_einsum.log_viterbi_einsum_forward
)
