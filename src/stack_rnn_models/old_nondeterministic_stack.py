import torch
from torch_semiring_einsum import compile_equation

import attr

from lib.pytorch_tools.set_slice import set_slice
from lib.semiring import log

class TooManyUpdates(ValueError):
    pass

@attr.s
class UpdateResult:
    j = attr.ib()
    gamma_j = attr.ib()
    alpha_j = attr.ib()

class NondeterministicStack:
    """An older version of the stack WFA that cannot pop the symbol on top of
    the initial bottom symbol."""

    def __init__(self, batch_size, num_states, stack_alphabet_size,
            sequence_length, normalize_reading, include_states_in_reading,
            block_size, dtype, device, semiring):
        """Implements the stack WFA data structures.

        Note that `sequence_length` corresponds to the length of the input to
        the NS-RNN model. It determines the maximum number of times `update`
        can be called; `update` can be called at most `sequence_length - 1`
        times."""
        super().__init__()
        B = self.batch_size = batch_size
        Q = self.num_states = num_states
        S = self.stack_alphabet_size = stack_alphabet_size
        n = self.sequence_length = sequence_length
        self.device = device
        self.semiring = semiring
        self.alpha = semiring.zeros((B, n, Q, S), dtype=dtype, device=device)
        if not normalize_reading:
            # If the stack reading is not going to be normalized, do not use
            # -inf for the 0 weights in the initial time step, but use a
            # really negative number. This avoids nans.
            semiring.get_tensor(self.alpha)[:, 0, :, :] = -1e10
        semiring.get_tensor(self.alpha)[:, 0, 0, 0] = semiring.one
        self.gamma = semiring.zeros((B, n-1, n-1, Q, S, Q, S), dtype=dtype, device=device)
        self.block_size = block_size
        self.normalize_reading = normalize_reading
        self.include_states_in_reading = include_states_in_reading
        self.j = 0
        self.alpha_j = semiring.on_tensor(self.alpha, lambda x: x[:, self.j])

    def update(self, push, repl, pop):
        # push : B x Q x S x Q x S
        # repl : B x Q x S x Q x S
        # pop : B x Q x S x Q
        if not (self.j + 1 < self.sequence_length):
            raise TooManyUpdates(
                f'attempting to compute timestep {self.j+1} (0-indexed), but '
                f'only {self.sequence_length} timesteps were allocated with '
                f'sequence_length')
        semiring = self.semiring
        block_size = self.block_size
        self.j = j = self.j + 1
        gamma_j = next_gamma_column(
            # B x (j-1) x (j-1) x Q x S x Q x S
            semiring.on_tensor(self.gamma, lambda x: x[:, :j-1, :(j)-1]),
            push,
            repl,
            pop,
            semiring,
            block_size
        )
        # This is just a long way of updating column j of gamma in-place.
        self.gamma = semiring.combine(
            [self.gamma, gamma_j],
            lambda args: set_slice(
                args[0],
                (slice(None), slice(None, j), (j)-1),
                args[1]))
        # alpha_j : B x Q x S
        self.alpha_j = next_alpha_column(
            # B x j x Q x S
            semiring.on_tensor(self.alpha, lambda x: x[:, :j]),
            # B x j x Q x S x Q x S
            gamma_j,
            semiring,
            block_size
        )
        # This is just a long way of updating entry j of alpha in-place.
        self.alpha = semiring.combine(
            [self.alpha, self.alpha_j],
            lambda args: set_slice(
                args[0],
                (slice(None), j),
                args[1]))
        return UpdateResult(j, gamma_j, self.alpha_j)

    def reading(self):
        # Return log P_j(r, y).
        # alpha[0...j] has already been computed.
        semiring = self.semiring
        # self.alpha_j : B x Q x S
        if self.include_states_in_reading:
            B = self.alpha_j.size(0)
            # result : B x (Q * S)
            result = semiring.on_tensor(self.alpha_j, lambda x: x.view(B, -1))
        else:
            # result : B x S
            result = semiring.sum(self.alpha_j, dim=1)
        if self.normalize_reading:
            # Using softmax, normalize the log-weights so they sum to 1.
            assert semiring is log
            result = torch.nn.functional.softmax(result, dim=1)
        return result

REPL_EQUATION = compile_equation('biqxsz,bszry->biqxry')
POP_EQUATION = compile_equation('bikqxty,bktysz,bszr->biqxry')

def next_gamma_column(gamma, push, repl, pop, semiring, block_size):
    # gamma : B x (T-1) x (T-1) x Q x S x Q x S
    # return : B x T x Q x S x Q x S
    T = semiring.get_tensor(gamma).size(1) + 1
    B, _, _, Q, S, *_ = semiring.get_tensor(gamma).size()
    # push : B x Q x S x Q x S
    # push_term : B x 1 x Q x S x Q x S
    push_term = semiring.on_tensor(push, lambda x: x[:, None])
    # repl_term : B x T-1 x Q x S x Q x S
    if T == 1:
        repl_term = semiring.primitive(
            semiring.get_tensor(gamma).new_empty(B, 0, Q, S, Q, S))
    else:
        repl_term = semiring.einsum(
            REPL_EQUATION,
            # B x T-1 x Q x S x Q x S
            semiring.on_tensor(gamma, lambda x: x[:, :, -1]),
            # B x Q x S x Q x S
            repl,
            block_size=block_size
        )
    # pop_term : B x T-2 x Q x S x Q x S
    if T <= 2:
        pop_term = semiring.primitive(
            semiring.get_tensor(gamma).new_empty(B, 0, Q, S, Q, S))
    else:
        pop_term = semiring.einsum(
            POP_EQUATION,
            # B x T-2 x T-2 x Q x S x Q x S
            semiring.on_tensor(gamma, lambda x: x[:, :-1, :-1]),
            # B x T-2 x Q x S x Q x S
            semiring.on_tensor(gamma, lambda x: x[:, 1:, -1]),
            # B x Q x S x Q
            pop,
            block_size=block_size
        )
    return semiring.combine([
        semiring.add(
            semiring.on_tensor(repl_term, lambda x: x[:, :-1]),
            pop_term
        ),
        semiring.on_tensor(repl_term, lambda x: x[:, -1:]),
        push_term
    ], lambda args: torch.cat(args, dim=1))

ALPHA_EQUATION = compile_equation('biqx,biqxry->bry')

def next_alpha_column(alpha, gamma_j, semiring, block_size):
    # alpha : B x T x Q x S
    # gamma_j : B x T x Q x S x Q x S
    # return : B x Q x S
    return semiring.einsum(
        ALPHA_EQUATION,
        alpha,
        gamma_j,
        block_size=block_size
    )
