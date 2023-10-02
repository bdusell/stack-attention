from collections.abc import Callable
from typing import Literal, Optional, Union

import torch
from torch_semiring_einsum import compile_equation, AutomaticBlockSize

from lib.pytorch_tools.set_slice import set_slice
from lib.semiring import Semiring, log
from .common import StackRNNBase
from .nondeterministic_stack import (
    NondeterministicStackRNN,
    NondeterministicStack,
    get_initial_gamma,
    get_initial_alpha,
    gamma_i_index,
    gamma_j_index,
    alpha_j_index
)
from .old_vector_nondeterministic_stack import (
    VectorNondeterministicStack as OldVectorNondeterministicStack
)

zeta_i_index = gamma_i_index
zeta_j_index = gamma_j_index

class VectorNondeterministicStackRNN(NondeterministicStackRNN):

    def __init__(self,
        input_size: int,
        num_states: int,
        stack_alphabet_size: int,
        stack_embedding_size: int,
        controller: Callable[[int], torch.nn.Module],
        controller_output_size: int,
        normalize_operations: bool=False,
        include_states_in_reading: bool=True,
        original_bottom_symbol_behavior: bool=False,
        bottom_vector: Literal['learned', 'one', 'zero']='learned',
        **kwargs
    ):
        Q = num_states
        S = stack_alphabet_size
        m = stack_embedding_size
        if not include_states_in_reading:
            raise ValueError('include_states_in_reading=False is not supported')
        super().__init__(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            controller=controller,
            controller_output_size=controller_output_size,
            normalize_operations=normalize_operations,
            original_bottom_symbol_behavior=original_bottom_symbol_behavior,
            stack_reading_size=Q * S * m,
            include_states_in_reading=include_states_in_reading,
            **kwargs
        )
        self.stack_embedding_size = stack_embedding_size
        self.pushed_vector_layer = torch.nn.Sequential(
            torch.nn.Linear(
                self.output_size(),
                stack_embedding_size
            ),
            torch.nn.LogSigmoid()
        )
        # This parameter is the learned embedding that always sits at the
        # bottom of the stack. It is the input to a sigmoid operation, so the
        # vector used in the stack will be in (0, 1).
        if original_bottom_symbol_behavior:
            if bottom_vector is not None:
                raise ValueError(
                    'if original_bottom_symbol_behavior=True, bottom_vector '
                    'must be None')
        else:
            self.bottom_vector_type = bottom_vector
            if bottom_vector == 'learned':
                self.bottom_vector = torch.nn.Parameter(torch.zeros((m,)))
            elif bottom_vector in ('one', 'zero'):
                pass
            else:
                raise ValueError(f'unknown bottom vector option: {bottom_vector!r}')

    def pushed_vector(self, hidden_state):
        return self.pushed_vector_layer(hidden_state)

    def get_bottom_vector(self, semiring):
        if self.bottom_vector_type == 'learned':
            if semiring is not log:
                raise NotImplementedError
            return torch.nn.functional.logsigmoid(self.bottom_vector)
        elif self.bottom_vector_type == 'one':
            tensor = next(self.parameters())
            return semiring.ones((self.stack_embedding_size,), like=tensor)
        elif self.bottom_vector_type == 'zero':
            return None
        else:
            raise ValueError

    def get_new_stack(self, batch_size, sequence_length, semiring, block_size):
        tensor = next(self.parameters())
        # If the stack reading is not included in the output, then the last
        # timestep is not needed.
        if not self.include_reading_in_output:
            sequence_length -= 1
        if not self.original_bottom_symbol_behavior:
            return get_vector_nondeterministic_stack(
                batch_size=batch_size,
                num_states=self.num_states,
                stack_alphabet_size=self.stack_alphabet_size,
                stack_embedding_size=self.stack_embedding_size,
                sequence_length=sequence_length,
                bottom_vector=self.get_bottom_vector(semiring),
                block_size=block_size,
                dtype=tensor.dtype,
                device=tensor.device,
                semiring=semiring
            )
        else:
            return OldVectorNondeterministicStack(
                batch_size=batch_size,
                num_states=self.num_states,
                stack_alphabet_size=self.stack_alphabet_size,
                stack_embedding_size=self.stack_embedding_size,
                sequence_length=sequence_length,
                block_size=block_size,
                dtype=tensor.dtype,
                device=tensor.device,
                semiring=semiring
            )

    class State(StackRNNBase.State):

        def compute_stack(self, hidden_state, stack):
            push, repl, pop = self.rnn.operation_log_scores(hidden_state)
            pushed_vector = self.rnn.pushed_vector(hidden_state)
            actions = (push, repl, pop, pushed_vector)
            stack.update(push, repl, pop, pushed_vector)
            return stack, actions

class VectorNondeterministicStack(NondeterministicStack):

    def __init__(self,
        gamma,
        alpha,
        alpha_j,
        zeta,
        zeta_j,
        timestep: int,
        sequence_length: Optional[int],
        block_size: Union[int, AutomaticBlockSize],
        semiring: Semiring
    ):
        super().__init__(
            gamma=gamma,
            alpha=alpha,
            alpha_j=alpha_j,
            timestep=timestep,
            sequence_length=sequence_length,
            include_states_in_reading=True,
            normalize_reading=True,
            block_size=block_size,
            semiring=semiring
        )
        self.zeta = zeta
        self.zeta_j = zeta_j
        self.stack_embedding_size = semiring.get_tensor(zeta).size(7)

    def update(self, push, repl, pop, pushed_vector):
        # push : B x Q x S x Q x S
        # repl : B x Q x S x Q x S
        # pop : B x Q x S x Q
        # pushed_vector : B x m
        # Update the self.gamma and self.alpha tables.
        result = super().update(push, repl, pop, return_gamma_prime=True)
        semiring = self.semiring
        block_size = self.block_size
        j = self.j
        # self.zeta_j : B x j+1 x Q x S x Q x S x m
        self.zeta_j = next_zeta_column(
            # B x j x j x Q x S x Q x S x m
            semiring.on_tensor(self.zeta, lambda x: x[:, :zeta_i_index(j-1), :zeta_j_index(j)]),
            # B x j-1 x Q x S x Q
            result.gamma_prime_j,
            push,
            repl,
            pushed_vector,
            semiring,
            block_size
        )
        result.gamma_prime_j = None
        # If the sequence length is unlimited, grow gamma by 1 first.
        if self.sequence_length is None:
            zeta_tensor = semiring.get_tensor(self.zeta)
            new_size = list(zeta_tensor.size())
            new_size[1] += 1
            new_size[2] += 1
            new_zeta = semiring.zeros(new_size, dtype=zeta_tensor.dtype, device=zeta_tensor.device)
            self.zeta = semiring.combine(
                [new_zeta, self.zeta],
                lambda args: set_slice(
                    args[0],
                    (slice(None), slice(None, zeta_i_index(j-1)), slice(None, zeta_j_index(j))),
                    args[1]))
        self.zeta = semiring.combine(
            [self.zeta, self.zeta_j],
            lambda args: set_slice(
                args[0],
                (slice(None), slice(None, zeta_i_index(j)), zeta_j_index(j)),
                args[1]))
        return result

    def reading(self):
        semiring = self.semiring
        # eta_j : B x Q x S x m
        eta_j = next_eta_column(
            semiring.on_tensor(self.alpha, lambda x: x[:, :alpha_j_index(self.j)]),
            self.zeta_j,
            semiring,
            self.block_size
        )
        return eta_to_reading(self.alpha_j, eta_j, semiring)

    def transform_tensors(self, func):
        return VectorNondeterministicStack(
            func(self.gamma),
            func(self.alpha),
            func(self.alpha_j),
            func(self.zeta),
            func(self.zeta_j),
            self.j,
            self.sequence_length,
            self.block_size,
            self.semiring
        )

def get_vector_nondeterministic_stack(
    batch_size: int,
    num_states: int,
    stack_alphabet_size: int,
    stack_embedding_size: int,
    sequence_length: Optional[int],
    bottom_vector: Optional[torch.Tensor],
    block_size: Union[int, AutomaticBlockSize],
    dtype: torch.dtype,
    device: torch.device,
    semiring: Semiring
) -> VectorNondeterministicStack:
    B = batch_size
    Q = num_states
    S = stack_alphabet_size
    n = sequence_length
    m = stack_embedding_size
    gamma = get_initial_gamma(B, Q, S, n, dtype, device, semiring)
    alpha, alpha_j = get_initial_alpha(B, Q, S, n, dtype, device, semiring)
    zeta, zeta_j = get_initial_zeta(B, Q, S, n, m, bottom_vector, dtype, device, semiring)
    return VectorNondeterministicStack(
        gamma=gamma,
        alpha=alpha,
        alpha_j=alpha_j,
        zeta=zeta,
        zeta_j=zeta_j,
        timestep=0,
        sequence_length=sequence_length,
        block_size=block_size,
        semiring=semiring
    )

def get_initial_zeta(B, Q, S, n, m, bottom_vector, dtype, device, semiring):
    # zeta[:, i+1, j, q, x, r, y] contains the value of
    # $\zeta[i \rightarrow j][q, x \rightarrow r, y]$ for 0 <= j <= n
    # and -1 <= i <= j-1.
    # So, the size of zeta is n+1 x n+1.
    # If the sequence length is unlimited, set the size to 0 at first.
    if n is None:
        n = 0
    zeta = semiring.zeros((B, n+1, n+1, Q, S, Q, S, m), dtype=dtype, device=device)
    # Initialize $\zeta[-1 \rightarrow 0]$ to the (possibly learned)
    # bottom vector. If bottom_vector is None, then set the bottom vector
    # to zero.
    if bottom_vector is not None:
        zeta = semiring.combine(
            [zeta, bottom_vector],
            lambda args: set_slice(
                args[0],
                (slice(None), zeta_i_index(-1), zeta_j_index(0), 0, 0, 0, 0),
                args[1]))
    zeta_j = semiring.on_tensor(
        zeta,
        lambda x: x[:, :zeta_i_index(0), zeta_j_index(0)])
    return zeta, zeta_j

ZETA_REPL_EQUATION = compile_equation('biqxszm,bszry->biqxrym')
ZETA_POP_EQUATION = compile_equation('bikqxtym,bktyr->biqxrym')

def next_zeta_column(zeta, gamma_prime_j, push, repl, pushed_vector, semiring,
        block_size):
    # zeta : B x T-1 x T-1 x Q x S x Q x S x m
    # gamma_prime_j : B x T-2 x Q x S x Q
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
            block_size=block_size,
            **(dict(grad_of_neg_inf=0.0) if semiring is log else {})
        )
    # pop_term : B x T-2 x Q x S x Q x S x m
    if T <= 2:
        pop_term = semiring.primitive(
            semiring.get_tensor(zeta).new_empty(B, 0, Q, S, Q, S, m))
    else:
        pop_term = semiring.einsum(
            ZETA_POP_EQUATION,
            # B x T-2 x T-2 x Q x S x Q x S x m
            semiring.on_tensor(zeta, lambda x: x[:, :-1, :-1]),
            # B x T-2 x Q x S x Q
            gamma_prime_j,
            block_size=block_size,
            **(dict(grad_of_neg_inf=0.0) if semiring is log else {})
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

def eta_to_reading(alpha_j, eta_j, semiring):
    assert semiring is log
    # alpha_j : B x Q x S
    # eta_j : B x Q x S x m
    # denom : B
    denom = semiring.sum(alpha_j, dim=(1, 2))
    # Divide (in log space) eta by the sum over alpha, then take the exp
    # to get back to real space. Finally, flatten the dimensions.
    B = eta_j.size(0)
    return torch.exp(eta_j - denom[:, None, None, None]).view(B, -1)
