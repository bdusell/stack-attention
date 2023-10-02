import torch
from torch_semiring_einsum import compile_equation

from lib.pytorch_tools.set_slice import set_slice
from lib.semiring import log
from .common import StackRNNBase
from .nondeterministic_stack import (
    NondeterministicStackRNN,
    NondeterministicStack
)
from .old_nondeterministic_stack import NondeterministicStack

class VectorNondeterministicStack(NondeterministicStack):

    def __init__(self, batch_size, num_states, stack_alphabet_size,
            stack_embedding_size, sequence_length, block_size, dtype, device,
            semiring):
        super().__init__(
            batch_size=batch_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            sequence_length=sequence_length,
            normalize_reading=True,
            include_states_in_reading=True,
            block_size=block_size,
            dtype=dtype,
            device=device,
            semiring=semiring
        )
        m = self.stack_embedding_size = stack_embedding_size
        B = self.batch_size
        Q = self.num_states
        S = self.stack_alphabet_size
        n = self.sequence_length
        self.zeta = semiring.zeros((B, n-1, n-1, Q, S, Q, S, m), dtype=dtype, device=device)
        self.zeta_j = None

    def update(self, push, repl, pop, pushed_vector):
        # push : B x Q x S x Q x S
        # repl : B x Q x S x Q x S
        # pop : B x Q x S x Q
        # pushed_vector : B x m
        # Update the self.gamma and self.alpha tables.
        super().update(push, repl, pop)
        semiring = self.semiring
        block_size = self.block_size
        j = self.j
        self.zeta_j = next_zeta_column(
            # B x j-1 x j-1 x Q x S x Q x S
            semiring.on_tensor(self.gamma, lambda x: x[:, :j-1, :j-1]),
            # B x j-1 x j-1 x Q x S x Q x S x m
            semiring.on_tensor(self.zeta, lambda x: x[:, :j-1, :j-1]),
            push,
            repl,
            pop,
            pushed_vector,
            semiring,
            block_size
        )
        self.zeta = semiring.combine(
            [self.zeta, self.zeta_j],
            lambda args: set_slice(
                args[0],
                (slice(None), slice(None, j), (j)-1),
                args[1]))

    def reading(self):
        if self.j == 0:
            B, _, _, Q, S, _, _, m = self.zeta.size()
            return self.zeta.new_zeros((B, Q * S * m))
        else:
            semiring = self.semiring
            # eta_j : B x Q x S x m
            eta_j = next_eta_column(
                semiring.on_tensor(self.alpha, lambda x: x[:, :self.j]),
                self.zeta_j,
                semiring,
                self.block_size
            )
            # self.alpha_j : B x Q x S
            # denom : B
            denom = semiring.sum(self.alpha_j, dim=(1, 2))
            assert semiring is log
            # Divide (in log space) eta by the sum over alpha, then take the exp
            # to get back to real space. Finally, flatten the dimensions.
            B = eta_j.size(0)
            return torch.exp(eta_j - denom[:, None, None, None]).view(B, -1)

ZETA_REPL_EQUATION = compile_equation('biqxszm,bszry->biqxrym')
ZETA_POP_EQUATION = compile_equation('bikqxtym,bktysz,bszr->biqxrym')

def next_zeta_column(gamma, zeta, push, repl, pop, pushed_vector, semiring,
        block_size):
    # gamma : B x T-1 x T-1 x Q x S x Q x S
    # zeta : B x T-1 x T-1 x Q x S x Q x S x m
    # return : B x T x Q x S x Q x S x m
    T = semiring.get_tensor(zeta).size(1) + 1
    B, _, _, Q, S, _, _, m = semiring.get_tensor(zeta).size()
    # push : B x Q x S x Q x S
    # pushed_vector : B x m
    # push_term : B x 1 x Q x S x Q x S x m
    push_term = semiring.on_tensor(
        # B x Q x S x Q x S x m
        semiring.multiply(
            # B x Q x S x Q x S x 1
            semiring.on_tensor(push, lambda x: x[:, :, :, :, :, None]),
            # B x 1 x 1 x 1 x 1 x m
            semiring.on_tensor(pushed_vector, lambda x: x[:, None, None, None, None, :])
        ),
        lambda x: x[:, None]
    )
    # repl_term : B x T-1 x Q x S x Q x S x m
    if T == 1:
        repl_term = semiring.primitive(
            semiring.get_tensor(zeta).new_empty(B, 0, Q, S, Q, S, m))
    else:
        repl_term = semiring.einsum(
            ZETA_REPL_EQUATION,
            # B x T-1 x Q x S x Q x S x m
            semiring.on_tensor(zeta, lambda x: x[:, :, -1]),
            # B x Q x S x Q x S
            repl,
            block_size=block_size
        )
    # pop_term : B x T-2 x Q x S x Q x S x m
    if T <= 2:
        B, _, _, Q, S, _, _, m = semiring.get_tensor(zeta).size()
        pop_term = semiring.primitive(
            semiring.get_tensor(gamma).new_empty(B, 0, Q, S, Q, S, m))
    else:
        pop_term = semiring.einsum(
            ZETA_POP_EQUATION,
            # B x T-2 x T-2 x Q x S x Q x S x m
            semiring.on_tensor(zeta, lambda x: x[:, :-1, :-1]),
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

ETA_EQUATION = compile_equation('biqx,biqxrym->brym')

def next_eta_column(alpha, zeta_j, semiring, block_size):
    # alpha : B x T x Q x S
    # zeta_j : B x T x Q x S x Q x S x m
    # return : B x Q x S x m
    return semiring.einsum(
        ETA_EQUATION,
        alpha,
        zeta_j,
        block_size=block_size
    )
