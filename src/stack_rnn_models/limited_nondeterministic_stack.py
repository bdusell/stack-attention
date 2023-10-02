import attr
import torch
from torch_semiring_einsum import compile_equation

from torch_unidirectional import Unidirectional
from lib.pytorch_tools.set_slice import set_slice
from lib.semiring import log, log_viterbi
from lib.data_structures.linked_list import LinkedList
from .nondeterministic_stack import (
    NondeterministicStackRNN,
    TooManyUpdates,
    UpdateResult,
    next_alpha_column,
    next_gamma_column,
    ensure_not_negative,
    Operation,
    PushOperation,
    ReplaceOperation,
    PopOperation
)

class LimitedNondeterministicStackRNN(NondeterministicStackRNN):

    def __init__(self, input_size, num_states, stack_alphabet_size,
            window_size, controller, normalize_operations=False,
            include_states_in_reading=True, **kwargs):
        super().__init__(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            controller=controller,
            normalize_operations=normalize_operations,
            include_states_in_reading=include_states_in_reading,
            **kwargs
        )
        self.window_size = window_size

    # Use a separate type for the initial state used with `initial_state()`,
    # `initial_state=` and `return_state=` that contains only the chunks of
    # gamma and alpha that need to be passed between batches.
    @attr.s
    class InitialState(Unidirectional.State):

        state = attr.ib()
        stack = attr.ib()

        def detach(self):
            return type(self)(self.state.detach(), self.stack.detach())

        def batch_size(self):
            return self.state.batch_size()

        def slice_batch(self, s):
            return type(self)(self.state.slice_batch(s), self.stack.slice_batch(s))

    class InitialStackState(Unidirectional.State):

        def __init__(self, gamma, alpha, semiring):
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha
            self.semiring = semiring

        def transform_tensors(self, func):
            # This implements detach() and slice_batch().
            return type(self)(
                self.semiring.on_tensor(self.gamma, func),
                self.semiring.on_tensor(self.alpha, func),
                self.semiring
            )

        def batch_size(self):
            return self.gamma.size(0)

    def forward(self, input_sequence, block_size, initial_state=None,
            return_state=False, include_first=True, return_actions=False,
            return_readings=False):
        semiring = log
        batch_size = input_sequence.size(0)
        sequence_length = input_sequence.size(1)
        if initial_state is None:
            initial_state = self.initial_state(batch_size, semiring)
        else:
            if not isinstance(initial_state, self.InitialState):
                raise TypeError
        # Convert initial_state from the InitialState type to State.
        stack_kwargs = dict(
            sequence_length=sequence_length,
            semiring=semiring,
            block_size=block_size
        )
        if initial_state.stack is None:
            stack = None
            stack_args = []
        else:
            stack = self.get_new_stack(
                initial_stack_state=initial_state.stack,
                batch_size=batch_size,
                **stack_kwargs
            )
            stack_args = None
            stack_kwargs = None
        state = self.State(
            rnn=self,
            hidden_state=initial_state.state,
            previous_stack=stack,
            return_actions=return_actions,
            # It might be helpful to grab this somehow from initial_state,
            # but if include_first is False, this won't be used anyway.
            previous_actions=None,
            return_readings=return_readings,
            previous_reading=None,
            stack_args=stack_args,
            stack_kwargs=stack_kwargs
        )
        result = super(NondeterministicStackRNN, self).forward(
            input_sequence,
            initial_state=state,
            return_state=return_state,
            include_first=include_first
        )
        if return_state:
            output, last_state = result
            stack = last_state.previous_stack
            last_state = self.InitialState(
                last_state.hidden_state,
                self.get_forwarded_stack_state(stack, semiring)
            )
            result = output, last_state
        return result

    def get_forwarded_stack_state(self, stack, semiring):
        # Convert the last state from the State type to the InitialState type.
        # Save some slices of gamma and alpha to forward to the next batch.
        D = self.window_size
        return self.InitialStackState(
            gamma=semiring.on_tensor(stack.gamma, lambda x: x[:, -(D-1):, -(D-1):]),
            alpha=semiring.on_tensor(stack.alpha, lambda x: x[:, -D:]),
            semiring=semiring
        )

    def initial_state(self, batch_size, semiring=log):
        return self.InitialState(
            self.controller.initial_state(batch_size),
            None
        )

    def get_new_viterbi_stack(self, batch_size, sequence_length, semiring,
            block_size, initial_stack_state=None):
        tensor = next(self.parameters())
        return LimitedNondeterministicStack(
            batch_size=batch_size,
            num_states=self.num_states,
            stack_alphabet_size=self.stack_alphabet_size,
            sequence_length=sequence_length,
            window_size=self.window_size,
            include_states_in_reading=self.include_states_in_reading,
            initial_state=initial_stack_state,
            semiring=semiring,
            block_size=block_size,
            dtype=tensor.dtype,
            device=tensor.device
        )

    def get_viterbi_decoder(self, alpha_columns, gamma_j_nodes, alpha_j_nodes):
        return LimitedViterbiDecoder(
            alpha_columns,
            gamma_j_nodes,
            alpha_j_nodes,
            self.window_size
        )

class LimitedNondeterministicStack:

    def __init__(self, batch_size, num_states, stack_alphabet_size,
            sequence_length, window_size, include_states_in_reading,
            initial_state, semiring, block_size, dtype, device):

        if not (window_size >= 1):
            raise ValueError('window_size must be at least 1')

        # Let j_1 be the first time step for which we will be computing gamma
        # and alpha, that is, gamma[i, j_1] and alpha[j_1].
        # We will be computing gamma and alpha for j = j_1 through j_1 + T - 1.
        # Since i >= j - D and i <= j - 1, computing gamma[i, j_1] only
        # requires computing gamma for i = j_1 - D through j_1 - 1.
        # Computing gamma[i -> j_1] for i = j_1 - D through j_1 - 1 only
        # requires the values of gamma[i, j] for j_1 - D + 1 <= j <= j_1 - 1.
        # Computing alpha[j_1] only requires alpha[j] for j = j_1 - D through
        # j_1 - 1 and gamma[i -> j_1] for i = j_1 - D through j_1 - 1.
        # This means that the initial stack state consists of:
        # (1) a square of gamma for gamma[j_1 - D + 1:j_1, j_1 - D + 1:j_1]
        # (2) a slice of alpha for alpha[j_1 - D:j_1].
        super().__init__()
        B = self.batch_size = batch_size
        Q = self.num_states = num_states
        S = self.stack_alphabet_size = stack_alphabet_size
        T = self.sequence_length = sequence_length
        D = self.window_size = window_size
        self.include_states_in_reading = include_states_in_reading
        self.semiring = semiring
        self.block_size = block_size
        self.device = device

        # Verify that pieces of alpha and gamma passed in from a previous batch
        # have the correct dimensions.
        if initial_state is not None:
            if semiring.get_tensor(initial_state.gamma).size() != (B, D-1, D-1, Q, S, Q, S):
                raise ValueError
            if semiring.get_tensor(initial_state.alpha).size() != (B, D, Q, S):
                raise ValueError

        # Initialize the current timestep to -1. It is incremented at the
        # beginning of update(), which computes column j of gamma. A subsequent
        # call to reading() should return entry j of alpha.
        # j is relative to the beginning of the current chunk and is reset at
        # the beginning of every chunk. For the first chunk, j = 0 corresponds
        # to timestep 1.
        self.j = -1

        self.alpha = semiring.zeros((B, D+T, Q, S), dtype=dtype, device=device)
        self.gamma = semiring.zeros((B, T+D-1, T+D-1, Q, S, Q, S), dtype=dtype, device=device)

        if initial_state is None:
            # If no initial state was given, automatically run the first
            # timestep, which populates the first entry of alpha. The stack is
            # in an invalid state otherwise.
            self._first_update()
        else:
            # Initialize alpha and gamma with the pieces passed in from the
            # previous batch.
            semiring.get_tensor(self.alpha)[:, :alpha_j_index(D, 0)] = semiring.get_tensor(initial_state.alpha)
            semiring.get_tensor(self.gamma)[:, :gamma_i_index(D, -1), :gamma_j_index(D, 0)] = semiring.get_tensor(initial_state.gamma)

    def _first_update(self):
        if self.j != -1:
            raise ValueError
        j = self.j = self.j + 1
        D = self.window_size
        semiring = self.semiring
        # Note that j is relative to the beginning of the chunk. For the first
        # chunk, j corresponds to timestep j (they are the same).
        semiring.get_tensor(self.gamma)[:, gamma_i_index(D, -1), gamma_j_index(D, 0), 0, 0, 0, 0] = semiring.one
        # alpha[-1] and alpha[0] are both initialized here.
        semiring.get_tensor(self.alpha)[:, (alpha_j_index(D, -1), alpha_j_index(D, 0)), 0, 0] = semiring.one

    def update(self, push, repl, pop, return_gamma_prime=False):
        # push : B x Q x S x Q x S
        # repl : B x Q x S x Q x S
        # pop : B x Q x S x Q
        if not (self.j + 1 < self.sequence_length):
            raise TooManyUpdates(
                f'attempting to compute timestep {self.j+1} (0-indexed), but '
                f'only {self.sequence_length} timesteps were allocated with '
                f'sequence_length')
        semiring = self.semiring
        D = self.window_size
        block_size = self.block_size
        # j is initialized to -1. The first call to update() will compute
        # column 0 of gamma and alpha.
        self.j = j = self.j + 1
        # gamma_j : B x D x Q x S x Q x S
        gamma_j, gamma_prime_j = next_gamma_column(
            # B x D-1 x D-1 x Q x S x Q x S
            semiring.on_tensor(self.gamma, lambda x: x[
                :,
                gamma_i_index(D, j-1-(D-1)):gamma_i_index(D, j-1),
                gamma_j_index(D, j-(D-1)):gamma_j_index(D, j)
            ]),
            push,
            repl,
            pop,
            semiring,
            block_size,
            return_gamma_prime,
            gamma_prime_zero_grad=True
        )
        # This is just a long way of updating column j of gamma in-place.
        self.gamma = semiring.combine(
            [self.gamma, gamma_j],
            lambda args: set_slice(
                args[0],
                (
                    slice(None),
                    slice(gamma_i_index(D, j-D), gamma_i_index(D, j)),
                    gamma_j_index(D, j)
                ),
                args[1]))
        # alpha_j : B x Q x S
        alpha_j = next_alpha_column(
            # B x D x Q x S
            semiring.on_tensor(self.alpha, lambda x: x[:, alpha_j_index(D, j-D):alpha_j_index(D, j)]),
            # B x D x Q x S x Q x S
            gamma_j,
            semiring,
            block_size
        )
        # This is just a long way of updating entry j of alpha in-place.
        self.alpha = semiring.combine(
            [self.alpha, alpha_j],
            lambda args: set_slice(
                args[0],
                (slice(None), alpha_j_index(D, j)),
                args[1]))
        return UpdateResult(j, gamma_j, alpha_j, gamma_prime_j)

    def reading(self):
        # Return log P_j(r, y).
        # alpha[0...j] has already been computed.
        semiring = self.semiring
        D = self.window_size
        # alpha_j : B x Q x S
        alpha_j = self.alpha[:, alpha_j_index(D, self.j)]
        if self.include_states_in_reading:
            B = alpha_j.size(0)
            # result : B x (Q * S)
            result = semiring.on_tensor(alpha_j, lambda x: x.view(B, -1))
        else:
            # result : B x S
            result = semiring.sum(alpha_j, dim=1)
        # Using softmax, normalize the log-weights so they sum to 1.
        assert semiring is log
        return torch.nn.functional.softmax(result, dim=1)

def gamma_i_index(D, i):
    # Note that i is relative to the beginning of the current chunk.
    return ensure_not_negative(D + i)

def gamma_j_index(D, j):
    # Note that j is the timestep relative to the beginning of the current
    # chunk. For the first chunk, j = 0 is timestep 1, not 0.
    return ensure_not_negative(D - 1 + j)

def alpha_j_index(D, j):
    return ensure_not_negative(D + j)

class LimitedViterbiDecoder:

    def __init__(self, alpha_columns, gamma_j_nodes, alpha_j_nodes, window_size):
        self.alpha_columns = alpha_columns
        self.gamma_j_nodes = gamma_j_nodes
        self.alpha_j_nodes = alpha_j_nodes
        self.window_size = window_size

    def decode_timestep(self, j):
        """Return the best path leading up to the reading at timestep j.

        Timesteps are 0-indexed, where j = 0 is the first stack reading
        computed from the base case for alpha, and j = n-1 is the last valid
        timestep, corresponding to the stack reading just before the last
        output. The Viterbi path leading up to timestep j is always of length
        j."""
        if not (0 <= j < self.sequence_length):
            raise ValueError(f'timestep ({j}) must be in [0, {self.sequence_length})')
        # Sum over states, then stack symbols.
        alpha_j_sum_scores, alpha_j_sum_node = \
            log_viterbi.sum(log_viterbi.sum(self.get_alpha_j(j), dim=1), dim=1)
        batch_size = alpha_j_sum_scores.size(0)
        paths = [
            self.decode_alpha_j_sum(alpha_j_sum_node, b, j)
            for b in range(batch_size)
        ]
        return paths, alpha_j_sum_scores

    def decode_alpha_j_sum(self, alpha_j_sum_node, b, j):
        y = alpha_j_sum_node.backpointers[b]
        alpha_j_sum_states_node, = alpha_j_sum_node.children
        r = alpha_j_sum_states_node.backpointers[b, y]
        return self.decode_alpha_j(b, j, r, y)

    def decode_alpha_j(self, b, j, r, y):
        if j > 0:
            alpha_j_node = self.get_alpha_j_node(j)
            relative_i, q, x = alpha_j_node.backpointers[b, r, y]
            i = j - self.window_size + relative_i.item()
            # Recurse on alpha[i] and gamma[i, j]
            alpha_path = self.decode_alpha_j(b, i, q, x)
            gamma_path = self.decode_gamma_j(b, i, j, q, x, r, y)
            path = alpha_path
            path.extend(gamma_path)
            return path
        elif j == 0:
            return LinkedList([])
        else:
            raise ValueError(f'logic error: invalid value for j ({j})')

    def decode_gamma_j(self, b, i, j, q, x, r, y):
        gamma_j_node = self.get_gamma_j_node(j)
        repl_pop_node, repl_node, push_node = gamma_j_node.children
        if i < j - self.window_size:
            raise ValueError
        elif i < j-2:
            relative_i = self.get_relative_i(i, j)
            is_pop = repl_pop_node.backpointers[b, relative_i, q, x, r, y].item()
            repl_node, pop_node = repl_pop_node.children
            if is_pop:
                return self.decode_pop(pop_node, b, i, j, q, x, r, y)
            else:
                return self.decode_repl(repl_node, b, i, j, q, x, r, y)
        elif i == j-2:
            return self.decode_repl(repl_node, b, i, j, q, x, r, y)
        elif i == j-1:
            return LinkedList([PushOperation(r.item(), y.item())])
        else:
            raise ValueError

    def decode_repl(self, repl_node, b, i, j, q, x, r, y):
        relative_i = self.get_relative_i(i, j)
        s, z = repl_node.backpointers[b, relative_i, q, x, r, y]
        path = self.decode_gamma_j(b, i, j-1, q, x, s, z)
        path.append(ReplaceOperation(r.item(), y.item()))
        return path

    def decode_pop(self, pop_node, b, i, j, q, x, r, y):
        relative_i = self.get_relative_i(i, j)
        relative_k, t = pop_node.backpointers[b, relative_i, q, x, r, y]
        gamma_1_node, gamma_prime_node = pop_node.children
        s, z = gamma_prime_node.backpointers[b, relative_k, t, y, r]
        k = j - (self.window_size - 1) + relative_k.item()
        gamma_1_path = self.decode_gamma_j(b, i, k, q, x, t, y)
        gamma_2_path = self.decode_gamma_j(b, k, j-1, t, y, s, z)
        path = gamma_1_path
        path.extend(gamma_2_path)
        path.append(PopOperation(r.item()))
        return path

    def get_alpha_j(self, j):
        # Let 0 be the first time step, where alpha[0] is the set of initial
        # weights where only alpha[0][0, 0] is set to 1.
        # self.alpha_columns[0] is actually alpha[1], so we need to adjust the
        # index accordingly.
        return self.alpha_columns[j-1]

    def get_alpha_j_node(self, j):
        return self.alpha_j_nodes[j-1]

    def get_gamma_j_node(self, j):
        # Return the node for computing all gamma entries of the form
        # gamma[i, j].
        return self.gamma_j_nodes[j-1]

    def get_relative_i(self, i, j):
        return i - (j - self.window_size)

    @property
    def sequence_length(self):
        return len(self.alpha_columns) + 1
