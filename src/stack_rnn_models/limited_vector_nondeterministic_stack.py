import attr
import torch
from torch_semiring_einsum import compile_equation

from torch_unidirectional import Unidirectional
from lib.pytorch_tools.set_slice import set_slice
from lib.semiring import log
from .nondeterministic_stack import (
    NondeterministicStackRNN
)
from .limited_nondeterministic_stack import (
    LimitedNondeterministicStackRNN,
    LimitedNondeterministicStack,
    gamma_i_index,
    gamma_j_index,
    alpha_j_index
)
from .vector_nondeterministic_stack import (
    VectorNondeterministicStackRNN,
    next_zeta_column,
    next_eta_column,
    eta_to_reading
)

zeta_i_index = gamma_i_index
zeta_j_index = gamma_j_index

class LimitedVectorNondeterministicStackRNN(
    LimitedNondeterministicStackRNN,
    VectorNondeterministicStackRNN):

    def __init__(self, input_size, num_states, stack_alphabet_size,
            stack_embedding_size, window_size, controller,
            normalize_operations=False, bottom_vector='learned'):
        super().__init__(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            stack_embedding_size=stack_embedding_size,
            window_size=window_size,
            controller=controller,
            normalize_operations=normalize_operations,
            include_states_in_reading=True
        )
        self.window_size = window_size

    InitialState = LimitedNondeterministicStackRNN.InitialState

    class InitialStackState(LimitedNondeterministicStackRNN.InitialStackState):

        def __init__(self, gamma, zeta, alpha, semiring):
            super().__init__(gamma, alpha, semiring)
            self.zeta = zeta

        def transform_tensors(self, func):
            # This implements detach() and slice_batch().
            return type(self)(
                self.semiring.on_tensor(self.gamma, func),
                self.semiring.on_tensor(self.zeta, func),
                self.semiring.on_tensor(self.alpha, func),
                self.semiring
            )

    def get_forwarded_stack_state(self, stack, semiring):
        D = self.window_size
        return self.InitialStackState(
            gamma=semiring.on_tensor(stack.gamma, lambda x: x[:, -(D-1):, -(D-1):]),
            zeta=semiring.on_tensor(stack.zeta, lambda x: x[:, -(D-1):, -(D-1):]),
            alpha=semiring.on_tensor(stack.alpha, lambda x: x[:, -D:]),
            semiring=semiring
        )

    def get_new_stack(self, batch_size, sequence_length, semiring, block_size,
            initial_stack_state=None):
        tensor = next(self.parameters())
        if initial_stack_state is None:
            bottom_vector = self.get_bottom_vector(semiring)
        else:
            bottom_vector = None
        return LimitedVectorNondeterministicStack(
            batch_size=batch_size,
            num_states=self.num_states,
            stack_alphabet_size=self.stack_alphabet_size,
            stack_embedding_size=self.stack_embedding_size,
            sequence_length=sequence_length,
            window_size=self.window_size,
            bottom_vector=bottom_vector,
            initial_state=initial_stack_state,
            semiring=semiring,
            block_size=block_size,
            dtype=tensor.dtype,
            device=tensor.device
        )

class LimitedVectorNondeterministicStack(LimitedNondeterministicStack):

    def __init__(self, batch_size, num_states, stack_alphabet_size,
            stack_embedding_size, sequence_length, window_size, bottom_vector,
            initial_state, semiring, block_size, dtype, device):

        m = self.stack_embedding_size = stack_embedding_size
        B = batch_size
        Q = num_states
        S = stack_alphabet_size
        T = sequence_length
        D = window_size

        # Verify that the piece of zeta passed in from a previous batch has
        # the correct dimensions.
        if initial_state is not None:
            if semiring.get_tensor(initial_state.zeta).size() != (B, D-1, D-1, Q, S, Q, S, m):
                raise ValueError

        self.zeta = semiring.zeros((B, T+D-1, T+D-1, Q, S, Q, S, m), dtype=dtype, device=device)

        super().__init__(
            batch_size=batch_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            sequence_length=sequence_length,
            window_size=window_size,
            include_states_in_reading=True,
            initial_state=initial_state,
            semiring=semiring,
            block_size=block_size,
            dtype=dtype,
            device=device
        )

        if initial_state is None:
            # Initialize $\zeta[-1 \rightaarrow 0]$ to the (possibly learned)
            # bottom vector.
            self.zeta = semiring.combine(
                [self.zeta, bottom_vector],
                lambda args: set_slice(
                    args[0],
                    (slice(None), zeta_i_index(D, -1), zeta_j_index(D, 0), 0, 0, 0, 0),
                    args[1]))
        else:
            # Initialize zeta with the corner passed in from the previous batch.
            semiring.get_tensor(self.zeta)[
                :,
                :zeta_i_index(D, -1),
                :zeta_j_index(D, 0)
            ] = semiring.get_tensor(initial_state.zeta)

    def update(self, push, repl, pop, pushed_vector):
        # push : B x Q x S x Q x S
        # repl : B x Q x S x Q x S
        # pop : B x Q x S x Q
        # pushed_vector : B x m
        result = super().update(push, repl, pop, return_gamma_prime=True)
        D = self.window_size
        semiring = self.semiring
        block_size = self.block_size
        j = self.j
        # zeta_j : B x D x Q x S x Q x S x m
        zeta_j = next_zeta_column(
            # B x D-1 x D-1 x Q x S x Q x S x m
            semiring.on_tensor(self.zeta, lambda x: x[
                :,
                zeta_i_index(D, j-1-(D-1)):zeta_i_index(D, j-1),
                zeta_j_index(D, j-(D-1)):zeta_j_index(D, j)
            ]),
            # B x D-2 x Q x S x Q
            result.gamma_prime_j,
            push,
            repl,
            pushed_vector,
            semiring,
            block_size
        )
        result.gamma_prime_j = None
        self.zeta = semiring.combine(
            [self.zeta, zeta_j],
            lambda args: set_slice(
                args[0],
                (
                    slice(None),
                    slice(zeta_i_index(D, j-D), zeta_i_index(D, j)),
                    zeta_j_index(D, j)
                ),
                args[1]))
        return result

    def reading(self):
        # For the very first timestep when no vectors have been pushed, the
        # stack reading should be 0. This is treated as a special case in the
        # unlimited version, but here it just so happens that the formula for
        # the reading outputs 0, so it is not treated as a special case.
        semiring = self.semiring
        j = self.j
        D = self.window_size
        # eta_j : B x Q x S x m
        eta_j = next_eta_column(
            # B x D x Q x S
            semiring.on_tensor(self.alpha, lambda x: x[:, alpha_j_index(D, j-D):alpha_j_index(D, j)]),
            # B x D x Q x S x Q x S x m
            semiring.on_tensor(self.zeta, lambda x: x[:, zeta_i_index(D, j-D):zeta_i_index(D, j), zeta_j_index(D, j)]),
            semiring,
            self.block_size
        )
        # alpha_j : B x Q x S
        alpha_j = semiring.on_tensor(self.alpha, lambda x: x[:, alpha_j_index(D, j)])
        return eta_to_reading(alpha_j, eta_j, semiring)
