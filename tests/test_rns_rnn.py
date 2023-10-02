import itertools
import math
import unittest

import numpy
import torch

from torch_unidirectional import UnidirectionalLSTM
from stack_rnn_models.nondeterministic_stack import (
    NondeterministicStackRNN,
    get_nondeterministic_stack,
    TooManyUpdates,
    Operation,
    PushOperation,
    ReplaceOperation,
    PopOperation
)
from lib.semiring import real, log
from reference_util import recursively_stack

PUSH = 'push'
REPL = 'repl'
POP = 'pop'

def make_operations(stack, ops):
    assert stack.batch_size() == 1
    B = 1
    Q = stack.num_states
    S = stack.stack_alphabet_size
    device = stack.device
    assert stack.semiring is log
    semiring = stack.semiring
    push = semiring.zeros((Q, S, Q, S), device=device)
    repl = semiring.zeros((Q, S, Q, S), device=device)
    pop = semiring.zeros((Q, S, Q), device=device)
    for op in ops:
        op_type, *rest = op
        if op_type == PUSH:
            q, x, r, y, w = rest
            push[q, x, r, y] = math.log(w)
        elif op_type == REPL:
            q, x, r, y, w = rest
            repl[q, x, r, y] = math.log(w)
        elif op_type == POP:
            q, x, r, w = rest
            pop[q, x, r] = math.log(w)
        else:
            raise ValueError
    return push[None, ...], repl[None, ...], pop[None, ...]

def update_stack(stack, ops):
    push, repl, pop = make_operations(stack, ops)
    stack.update(push, repl, pop)

def make_reading_with_size(stack, entries, size):
    assert stack.batch_size() == 1
    device = stack.device
    reading = torch.zeros(size, device=device)
    for r, y, v in entries:
        reading[r, y] = v
    return reading[None, ...]

def make_reading(stack, entries):
    Q = stack.num_states
    S = stack.stack_alphabet_size
    return make_reading_with_size(stack, entries, (Q, S))

def get_reading(stack):
    B = stack.batch_size()
    Q = stack.num_states
    S = stack.stack_alphabet_size
    return stack.reading().view(B, Q, S)

class TestNondeterministicStackRNN(unittest.TestCase):

    def assert_is_finite(self, tensor, message=None):
        self.assertTrue(torch.all(torch.isfinite(tensor)).item(), message)

    def test_forward_and_backward(self):
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        hidden_units = 11
        sequence_length = 13
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = NondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            controller=controller,
            controller_output_size=hidden_units,
            normalize_operations=False,
            include_states_in_reading=True
        )
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1, generator=generator)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = torch.nn.MSELoss()
        input_tensor = torch.empty(batch_size, sequence_length - 1, input_size)
        input_tensor.uniform_(-1, 1, generator=generator)
        predicted_tensor = model(input_tensor, block_size=10)
        self.assertEqual(
            predicted_tensor.size(),
            (batch_size, sequence_length, hidden_units),
            'output does not have expected dimensions'
        )
        self.assert_is_finite(predicted_tensor)
        target_tensor = torch.empty(batch_size, sequence_length, hidden_units)
        target_tensor.uniform_(-1, 1, generator=generator)
        loss = criterion(predicted_tensor, target_tensor)
        optimizer.zero_grad()
        loss.backward()
        for name, p in model.named_parameters():
            self.assert_is_finite(p.grad, f'gradient for parameter {name} is not finite')
        optimizer.step()

    def test_against_reference(self):
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        hidden_units = 11
        sequence_length = 6
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = NondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            controller=controller,
            controller_output_size=hidden_units,
            normalize_operations=False,
            include_states_in_reading=True
        )
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1, generator=generator)
        reference_model = ReferenceNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            controller=controller,
            controller_output_size=hidden_units,
            normalize_operations=False,
            include_states_in_reading=True
        )
        for p, reference_p in itertools.zip_longest(model.parameters(), reference_model.parameters()):
            reference_p.data.copy_(p.data)
        input_tensor = torch.empty(batch_size, sequence_length - 1, input_size)
        input_tensor.uniform_(-1, 1, generator=generator)
        predicted_tensor = model(input_tensor, block_size=10)
        reference_predicted_tensor = reference_model(input_tensor)
        numpy.testing.assert_allclose(
            predicted_tensor.detach(),
            reference_predicted_tensor.detach(),
            rtol=1e-4,
            err_msg='output of model does not agree with reference implementation')
        criterion = torch.nn.MSELoss()
        target_tensor = torch.empty(batch_size, sequence_length, hidden_units)
        target_tensor.uniform_(-1, 1, generator=generator)
        loss = criterion(predicted_tensor, target_tensor)
        loss.backward()
        reference_loss = criterion(reference_predicted_tensor, target_tensor)
        reference_loss.backward()
        for p, reference_p in itertools.zip_longest(model.parameters(), reference_model.parameters()):
            numpy.testing.assert_allclose(
                p.grad,
                reference_p.grad,
                rtol=1e-3,
                err_msg='gradient does not agree with reference implementation')

    def test_return_actions_and_readings(self):
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        hidden_units = 11
        input_length = 5
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = NondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            controller=controller,
            controller_output_size=hidden_units,
            normalize_operations=False,
            include_states_in_reading=True
        )
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1, generator=generator)
        input_tensor = torch.empty(batch_size, input_length, input_size)
        input_tensor.uniform_(-1, 1, generator=generator)
        result = model(
            input_tensor,
            return_actions=True,
            return_readings=True,
            block_size=10
        )
        actions, readings = result.extra_outputs
        self.assertEqual(len(actions), input_length + 1)
        for action in (actions[0], actions[-1]):
            self.assertIsNone(action)
        for action in actions[1:-1]:
            self.assertIsNotNone(action)
        self.assertEqual(len(readings), input_length + 1)
        self.assertIsNone(readings[-1])
        for reading in readings[:-1]:
            self.assertIsNotNone(reading)

    def test_validate_sequence_length(self):
        B = 1
        Q = 2
        S = 3
        n = 4
        device = torch.device('cpu')
        semiring = log
        generator = torch.manual_seed(123)
        stack = get_nondeterministic_stack(
            batch_size=B,
            num_states=Q,
            stack_alphabet_size=S,
            sequence_length=n,
            include_states_in_reading=True,
            normalize_reading=True,
            block_size=10,
            dtype=torch.float32,
            device=device,
            semiring=semiring
        )
        def new_tensor(size):
            return torch.empty(size, device=device).uniform_(-10, 10, generator=generator)
        push = new_tensor((B, Q, S, Q, S))
        repl = new_tensor((B, Q, S, Q, S))
        pop = new_tensor((B, Q, S, Q))
        for i in range(n):
            stack.update(push, repl, pop)
        with self.assertRaises(TooManyUpdates):
            stack.update(push, repl, pop)

    def test_push_repl_push_pop(self):
        device = torch.device('cpu')
        semiring = log
        B = 1
        Q = 3
        S = 4
        stack = get_nondeterministic_stack(
            batch_size=B,
            num_states=Q,
            stack_alphabet_size=S,
            sequence_length=5,
            include_states_in_reading=True,
            normalize_reading=True,
            block_size=32,
            dtype=torch.float32,
            device=device,
            semiring=semiring
        )

        q0, q1, q2 = range(Q)

        # Stack: [ 0 ] ; q0 / 1.0
        expected_reading = make_reading(stack, [(q0, 0, 1.0)])
        numpy.testing.assert_allclose(get_reading(stack), expected_reading)

        update_stack(stack, [(PUSH, q0, 0, q1, 1, 0.5)])
        # Stack: [ 0, 1 ] ; q1 / 0.5
        expected_reading = make_reading(stack, [(q1, 1, 1.0)])
        numpy.testing.assert_allclose(get_reading(stack), expected_reading)

        update_stack(stack, [(REPL, q1, 1, q2, 2, 20.0)])
        # Stack: [ 0, 2 ] ; q2 / 10.0
        expected_reading = make_reading(stack, [(q2, 2, 1.0)])
        numpy.testing.assert_allclose(get_reading(stack), expected_reading)

        update_stack(stack, [(PUSH, q2, 2, q0, 3, 1.1)])
        # Stack: [ 0, 2, 3 ] ; q0 / 11.0
        expected_reading = make_reading(stack, [(q0, 3, 1.0)])
        numpy.testing.assert_allclose(get_reading(stack), expected_reading)

        update_stack(stack, [(POP, q0, 3, q1, 0.1)])
        # Stack: [ 0, 2 ] ; q1 / 1.1
        expected_reading = make_reading(stack, [(q1, 2, 1.0)])
        numpy.testing.assert_allclose(get_reading(stack), expected_reading)

    def test_push_repl_pop(self):
        device = torch.device('cpu')
        semiring = log
        B = 1
        Q = 3
        S = 4
        stack = get_nondeterministic_stack(
            batch_size=B,
            num_states=Q,
            stack_alphabet_size=S,
            sequence_length=4,
            include_states_in_reading=True,
            normalize_reading=True,
            block_size=32,
            dtype=torch.float32,
            device=device,
            semiring=semiring
        )

        q0, q1, q2 = range(Q)

        # Stack: [ 0 ] ; q0 / 1.0
        expected_reading = make_reading(stack, [(q0, 0, 1.0)])
        numpy.testing.assert_allclose(get_reading(stack), expected_reading)

        update_stack(stack, [(PUSH, q0, 0, q1, 1, 0.5)])
        # Stack: [ 0, 1 ] ; q1 / 0.5
        expected_reading = make_reading(stack, [(q1, 1, 1.0)])
        numpy.testing.assert_allclose(get_reading(stack), expected_reading)

        update_stack(stack, [(REPL, q1, 1, q2, 2, 20.0)])
        # Stack: [ 0, 2 ] ; q2 / 10.0
        expected_reading = make_reading(stack, [(q2, 2, 1.0)])
        numpy.testing.assert_allclose(get_reading(stack), expected_reading)

        update_stack(stack, [(POP, q2, 2, q1, 0.1)])
        # Stack: [ 0 ] ; q1 / 1.0
        expected_reading = make_reading(stack, [(q1, 0, 1.0)])
        numpy.testing.assert_allclose(get_reading(stack), expected_reading)

    def test_log_matches_real(self):
        # Test that the stack produces the same outputs and gradients in both
        # the real and log semirings.
        device = torch.device('cpu')
        B = batch_size = 5
        Q = num_states = 2
        S = stack_alphabet_size = 3
        n = sequence_length = 13
        generator = torch.manual_seed(123)
        real_stack = get_nondeterministic_stack(
            batch_size=batch_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            sequence_length=sequence_length,
            include_states_in_reading=True,
            normalize_reading=True,
            block_size=32,
            dtype=torch.float32,
            device=device,
            semiring=real
        )
        log_stack = get_nondeterministic_stack(
            batch_size=batch_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            sequence_length=sequence_length,
            include_states_in_reading=True,
            normalize_reading=True,
            block_size=32,
            dtype=torch.float32,
            device=device,
            semiring=log
        )
        def random_weights(size):
            return torch.nn.Parameter(
                torch.empty(size, device=device).uniform_(0.001, 5.0, generator=generator))
        real_params = real_stack_ops = [
            (random_weights((B, Q, S, Q, S)), random_weights((B, Q, S, Q, S)), random_weights((B, Q, S, Q)))
            for i in range(n-1)
        ]
        log_params = [
            tuple(torch.nn.Parameter(op.detach().clone()) for op in ops)
            for ops in real_stack_ops
        ]
        log_stack_ops = [
            tuple(torch.log(op) for op in ops)
            for ops in log_params
        ]
        def get_readings(stack, ops):
            result = [stack.reading()]
            for push, repl, pop in ops:
                stack.update(push, repl, pop)
                result.append(stack.reading())
            return result
        def run_backward(readings):
            # NOTE! It is very important not to include all elements of each
            # reading in the summation, because each reading sums to 1. If all
            # elements are summed, the loss function is constant, and the
            # gradient should be 0! In reality, the gradients will appear not
            # to match sometimes, because sometimes the gradient is not exactly
            # 0 due to rounding error. (I spent a lot of time debugging this.)
            # To get around this, we arbitrarily sum the first element of the
            # readings.
            torch.stack(readings)[:, :, 0].sum().backward()
        real_readings = get_readings(real_stack, real_stack_ops)
        run_backward(real_readings)
        log_readings = get_readings(log_stack, log_stack_ops)
        run_backward(log_readings)
        for i, (real_reading, log_reading) in enumerate(zip(real_readings, log_readings)):
            numpy.testing.assert_allclose(
                log_reading.detach(),
                real_reading.detach(),
                rtol=1e-5,
                err_msg=f'reading does not match at timestep {i}')
        for i, (real_params_i, log_params_i) in enumerate(zip(real_params, log_params)):
            for name, real_param, log_param in zip(['push', 'repl', 'pop'], real_params_i, log_params_i):
                if real_param.grad is None:
                    self.assertIsNone(log_param.grad)
                else:
                    numpy.testing.assert_allclose(
                        log_param.grad,
                        real_param.grad,
                        rtol=1e-1,
                        err_msg=f'gradient of {name} operations does not match at timestep {i}')

    def test_viterbi(self):
        # Test that the Viterbi algorithm runs without error.
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        hidden_units = 11
        sequence_length = 13
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = NondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            controller=controller,
            controller_output_size=hidden_units,
            normalize_operations=False,
            include_states_in_reading=True
        )
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1, generator=generator)
        input_tensor = torch.empty(batch_size, sequence_length, input_size)
        input_tensor.uniform_(-1, 1, generator=generator)
        model.eval()
        decoder = model.viterbi_decoder(input_tensor, block_size=10)
        for i in range(1, sequence_length):
            with self.subTest(f'j = {i}'):
                paths, scores = decoder.decode_timestep(i)
                for path in paths:
                    self.assertEqual(len(list(path)), i)
                    for operation in path:
                        self.assertIsInstance(operation, Operation)
                self.assertEqual(scores.size(), (batch_size,))

    def test_hardcoded_best_runs(self):
        # Test that the output of the Viterbi algorithm is correct on a
        # hard-coded example.
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        hidden_units = 11
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = NondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            controller=controller,
            controller_output_size=hidden_units,
            normalize_operations=False,
            include_states_in_reading=True
        )
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1, generator=generator)

        expected_run = [
            PushOperation(state_to=1, symbol=1),
            ReplaceOperation(state_to=0, symbol=2),
            PushOperation(state_to=1, symbol=0),
            PopOperation(state_to=0),
            ReplaceOperation(state_to=1, symbol=1),
            PushOperation(state_to=0, symbol=2)
        ]

        batch_size = 1
        B = batch_size
        Q = num_states
        S = stack_alphabet_size
        operation_weights = []
        for operation in expected_run:
            push = log.zeros((B, Q, S, Q, S))
            repl = log.zeros((B, Q, S, Q, S))
            pop = log.zeros((B, Q, S, Q))
            if type(operation) is PushOperation:
                push[:, :, :, operation.state_to, operation.symbol] = log.one
            elif type(operation) is ReplaceOperation:
                repl[:, :, :, operation.state_to, operation.symbol] = log.one
            elif type(operation) is PopOperation:
                pop[:, :, :, operation.state_to] = log.one
            operation_weights.append((push, repl, pop))

        model.eval()
        with torch.no_grad():
            decoder = model.viterbi_decoder_from_operation_weights(operation_weights, block_size=10)
            (path,), scores = decoder.decode_timestep(len(operation_weights))
        self.assertEqual(list(path), expected_run)
        self.assertEqual(scores.size(), (1,))
        numpy.testing.assert_allclose(scores, torch.tensor([0.0]))

class ReferenceNondeterministicStackRNN(NondeterministicStackRNN):

    def initial_stack(self, batch_size, stack_size, sequence_length):
        assert self.include_states_in_reading
        tensor = next(self.parameters())
        return ReferenceNondeterministicStack(
            batch_size,
            self.num_states,
            self.stack_alphabet_size,
            tensor.device
        )

class RealSemiring:

    @property
    def semiring(self):
        return real

    def normalize(self, x, dim):
        return x / x.sum(dim=dim, keepdim=True)

    def divide_to_real(self, a, b):
        return a / b

    def convert_operation_weights(self, x):
        return torch.exp(x)

class LogSemiring:

    @property
    def semiring(self):
        return log

    def normalize(self, x, dim):
        return torch.nn.functional.softmax(x, dim=dim)

    def divide_to_rela(self, a, b):
        return torch.exp(a - b)

    def convert_operation_weights(self, x):
        return x

def ensure_not_negative(x):
    if x < 0:
        raise ValueError
    return x

class ReferenceNondeterministicStack(RealSemiring):
    """This is a much slower but more obviously correct implementation of the
    stack WFA."""

    def __init__(self, B, Q, S, device):
        super().__init__()
        self.B = B
        self.Q = Q
        self.S = S
        self.device = device
        self.gamma = self.semiring.zeros((B, 1, 1, Q, S, Q, S), device=device)
        self.gamma[:, self.gamma_i_index(-1), self.gamma_j_index(0), 0, 0, 0, 0] = self.semiring.one
        self.alpha = self.semiring.zeros((B, 1, Q, S), device=device)
        self.alpha[:, self.alpha_j_index(-1), 0, 0] = self.semiring.one
        self.j = 0
        self.update_alpha()

    def gamma_i_index(self, i):
        return ensure_not_negative(i+1)

    def gamma_j_index(self, j):
        return ensure_not_negative(j)

    def alpha_j_index(self, j):
        return ensure_not_negative(j+1)

    def get(self, container, key):
        try:
            return container[key]
        except IndexError:
            return self.semiring.zeros((), device=self.device)

    def sum(self, tensors, size=()):
        if tensors:
            return self.semiring.sum(torch.stack(tensors), dim=0)
        else:
            if size is None:
                raise ValueError
            return self.semiring.zeros(size, device=self.device)

    def get_gamma(self, b, i, j, q, x, r, y):
        return self.get(self.gamma, (b, self.gamma_i_index(i), self.gamma_j_index(j), q, x, r, y))

    def get_alpha(self, b, j, r, y):
        return self.alpha[b, self.alpha_j_index(j), r, y]

    def update_gamma(self, push, repl, pop):
        B = self.B
        Q = self.Q
        S = self.S
        j = self.j
        push = self.convert_operation_weights(push)
        repl = self.convert_operation_weights(repl)
        pop = self.convert_operation_weights(pop)
        gamma_j = recursively_stack([
            [
                [
                    [
                        [
                            [
                                self.compute_gamma(push, repl, pop, b, i, j, q, x, r, y)
                                for y in range(S)
                            ]
                            for r in range(Q)
                        ]
                        for x in range(S)
                    ]
                    for q in range(Q)
                ]
                for i in range(-1, j-1+1)
            ]
            for b in range(B)
        ])
        self.gamma = torch.cat([
            torch.cat([
                self.gamma,
                self.semiring.zeros((B, 1, j, Q, S, Q, S), device=self.device)
            ], dim=1),
            gamma_j[:, :, None]
        ], dim=2)

    def compute_gamma(self, push, repl, pop, b, i, j, q, x, r, y):
        Q = self.Q
        S = self.S
        return self.sum([
            push[b, q, x, r, y] if i == j-1 else self.semiring.zeros((), device=self.device),
            self.sum([
                self.semiring.multiply(
                    self.get_gamma(b, i, j-1, q, x, s, z),
                    repl[b, s, z, r, y]
                )
                for s in range(Q)
                for z in range(S)
            ]),
            self.sum([
                self.semiring.multiply(
                    self.semiring.multiply(
                        self.get_gamma(b, i, k, q, x, t, y),
                        self.get_gamma(b, k, j-1, t, y, s, z)
                    ),
                    pop[b, s, z, r]
                )
                for k in range(i+1, j-2+1)
                for t in range(Q)
                for s in range(Q)
                for z in range(S)
            ])
        ])

    def update_alpha(self):
        B = self.B
        Q = self.Q
        S = self.S
        j = self.j
        alpha_j = recursively_stack([
            [
                [
                    self.sum([
                        self.semiring.multiply(
                            self.get_alpha(b, i, q, x),
                            self.get_gamma(b, i, j, q, x, r, y)
                        )
                        for i in range(-1, j-1+1)
                        for q in range(Q)
                        for x in range(S)
                    ])
                    for y in range(S)
                ]
                for r in range(Q)
            ]
            for b in range(B)
        ])
        self.alpha = torch.cat([
            self.alpha,
            alpha_j[:, None]
        ], dim=1)

    def update(self, push, repl, pop):
        self.j += 1
        self.update_gamma(push, repl, pop)
        self.update_alpha()

    def reading(self):
        B = self.B
        Q = self.Q
        S = self.S
        j = self.j
        return self.normalize(recursively_stack([
            [
                self.get_alpha(b, self.j, r, y)
                for r in range(Q)
                for y in range(S)
            ]
            for b in range(B)
        ]), dim=1)

if __name__ == '__main__':
    unittest.main()
