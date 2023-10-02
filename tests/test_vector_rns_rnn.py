import math
import unittest

import numpy
import pytest
import torch

from torch_unidirectional import UnidirectionalLSTM
from lib.semiring import log
from stack_rnn_models.nondeterministic_stack import TooManyUpdates
from stack_rnn_models.vector_nondeterministic_stack import (
    VectorNondeterministicStackRNN,
    get_vector_nondeterministic_stack
)
from test_rns_rnn import (
    make_operations,
    PUSH,
    REPL,
    POP,
    make_reading_with_size,
    ReferenceNondeterministicStack,
    ensure_not_negative
)
from reference_util import recursively_stack

def update_stack(stack, ops, pushed_vector):
    push, repl, pop = make_operations(stack, ops)
    pushed_vector = torch.log(pushed_vector)
    stack.update(push, repl, pop, pushed_vector[None, ...])

def make_reading(stack, entries):
    Q = stack.num_states
    S = stack.stack_alphabet_size
    m = stack.stack_embedding_size
    return make_reading_with_size(stack, entries, (Q, S, m))

def get_reading(stack):
    B = stack.batch_size()
    Q = stack.num_states
    S = stack.stack_alphabet_size
    m = stack.stack_embedding_size
    return stack.reading().view(B, Q, S, m)

class TestVectorRNSRNN(unittest.TestCase):

    def assert_is_finite(self, tensor, message=None):
        self.assertTrue(torch.all(torch.isfinite(tensor)).item(), message)

    def test_forward_and_backward(self):
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        stack_embedding_size = 4
        hidden_units = 11
        sequence_length = 13
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = VectorNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            stack_embedding_size=stack_embedding_size,
            controller=controller,
            controller_output_size=hidden_units,
            normalize_operations=False
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

    def test_single_run(self):
        device = torch.device('cpu')
        semiring = log
        B = 1
        Q = 3
        S = 4
        m = 2
        stack = get_vector_nondeterministic_stack(
            batch_size=B,
            num_states=Q,
            stack_alphabet_size=S,
            stack_embedding_size=m,
            sequence_length=4,
            bottom_vector=torch.full((m,), -math.inf),
            block_size=32,
            dtype=torch.float32,
            device=device,
            semiring=semiring
        )

        q0, q1, q2 = range(Q)
        v0 = torch.tensor([0.1, 0.1], device=device)
        v1 = torch.tensor([0.9, 0.1], device=device)
        v2 = torch.tensor([0.1, 0.9], device=device)

        # Stack: [ (0, 0) ] ; q0
        expected_reading = make_reading(stack, [])
        numpy.testing.assert_allclose(get_reading(stack), expected_reading)

        update_stack(stack, [(PUSH, q0, 0, q1, 1, 0.5)], v1)
        # Stack: [ (0, 0), (1, v1) ] ; q1
        expected_reading = make_reading(stack, [(q1, 1, v1)])
        numpy.testing.assert_allclose(get_reading(stack), expected_reading)

        update_stack(stack, [(REPL, q1, 1, q2, 2, 20.0)], v0)
        # Stack: [ (0, 0), (2, v1) ] ; q2
        expected_reading = make_reading(stack, [(q2, 2, v1)])
        numpy.testing.assert_allclose(get_reading(stack), expected_reading)

        update_stack(stack, [(PUSH, q2, 2, q0, 3, 1.1)], v2)
        # Stack: [ (0, 0), (2, v1), (3, v2) ] ; q0
        expected_reading = make_reading(stack, [(q0, 3, v2)])
        numpy.testing.assert_allclose(get_reading(stack), expected_reading)

        update_stack(stack, [(POP, q0, 3, q1, 0.1)], v0)
        # Stack: [ (0, 0), (2, v1) ] ; q1
        expected_reading = make_reading(stack, [(q1, 2, v1)])
        numpy.testing.assert_allclose(get_reading(stack), expected_reading)

    def test_two_runs(self):
        device = torch.device('cpu')
        semiring = log
        B = 1
        Q = 2
        S = 3
        m = 3
        stack = get_vector_nondeterministic_stack(
            batch_size=B,
            num_states=Q,
            stack_alphabet_size=S,
            stack_embedding_size=m,
            sequence_length=5,
            bottom_vector=torch.full((m,), -math.inf),
            block_size=32,
            dtype=torch.float32,
            device=device,
            semiring=semiring
        )

        q0, q1 = range(Q)
        v0 = torch.tensor([0.1, 0.1, 0.1], device=device)
        v1 = torch.tensor([0.9, 0.1, 0.1], device=device)
        v2 = torch.tensor([0.1, 0.9, 0.1], device=device)
        v3 = torch.tensor([0.1, 0.1, 0.9], device=device)

        # Stack: [ (0, 0) ] ; q0 / 1
        expected_reading = make_reading(stack, [])
        numpy.testing.assert_allclose(get_reading(stack), expected_reading)

        update_stack(stack, [(PUSH, q0, 0, q1, 1, 1)], v1)
        # Stack: [ (0, 0), (1, v1) ] ; q1 / 1
        expected_reading = make_reading(stack, [(q1, 1, v1)])
        numpy.testing.assert_allclose(get_reading(stack), expected_reading)

        update_stack(stack, [
            (REPL, q1, 1, q0, 2, 2),
            (PUSH, q1, 1, q1, 2, 1)
        ], v2)
        # Stack 1: [ (0, 0), (2, v1) ] ; q0 / 2
        # Stack 2: [ (0, 0), (1, v1), (2, v2) ] ; q1 / 1
        expected_reading = make_reading(stack, [
            (q0, 2, 2*v1/3),
            (q1, 2, v2/3)
        ])
        numpy.testing.assert_allclose(get_reading(stack), expected_reading, rtol=1e-6)

        update_stack(stack, [
            (PUSH, q0, 2, q0, 2, 2),
            (REPL, q1, 2, q1, 2, 1/2)
        ], v3)
        # Stack 1: [ (0, 0), (2, v1), (2, v3) ] ; q0 / 4
        # Stack 2: [ (0, 0), (1, v1), (2, v2) ] ; q1 / 1/2
        expected_reading = make_reading(stack, [
            (q0, 2, (4*v3)/(4+1/2)),
            (q1, 2, (1/2*v2)/(4+1/2))
        ])
        numpy.testing.assert_allclose(get_reading(stack), expected_reading)

        update_stack(stack, [
            (REPL, q0, 2, q0, 1, 1/5),
            (POP, q1, 2, q1, 3)
        ], v0)
        # Stack 1: [ (0, 0), (2, v1), (2, v3) ] ; q0 / 4/5
        # Stack 2: [ (0, 0), (1, v1) ] ; q1 / 3/2
        expected_reading = make_reading(stack, [
            (q0, 1, (4/5*v3)/(4/5+3/2)),
            (q1, 1, (3/2*v1)/(4/5+3/2))
        ])
        numpy.testing.assert_allclose(get_reading(stack), expected_reading, rtol=1e-6)

        update_stack(stack, [
            (POP, q0, 1, q0, 7),
            (REPL, q1, 1, q0, 2, 1)
        ], v0)
        # Stack 1: [ (0, 0), (2, v1) ] ; q0 / 28/5
        # Stack 2: [ (0, 0), (2, v1) ] ; q0 / 3/2
        expected_reading = make_reading(stack, [
            (q0, 2, v1)
        ])
        numpy.testing.assert_allclose(get_reading(stack), expected_reading, rtol=1e-6)

    def test_push_repl_pop(self):
        device = torch.device('cpu')
        semiring = log
        B = 1
        Q = 3
        S = 4
        m = 2
        stack = get_vector_nondeterministic_stack(
            batch_size=B,
            num_states=Q,
            stack_alphabet_size=S,
            stack_embedding_size=m,
            sequence_length=3,
            bottom_vector=torch.full((m,), -math.inf),
            block_size=32,
            dtype=torch.float32,
            device=device,
            semiring=semiring
        )
        v0 = torch.tensor([0.1, 0.1], device=device)
        v1 = torch.tensor([0.9, 0.1], device=device)

        q0, q1, q2 = range(Q)

        # Stack: [ (0, 0) ] ; q0 / 1.0
        expected_reading = make_reading(stack, [])
        numpy.testing.assert_allclose(get_reading(stack), expected_reading)

        update_stack(stack, [(PUSH, q0, 0, q1, 1, 0.5)], v1)
        # Stack: [ (0, 0), (1, v1) ] ; q1 / 0.5
        expected_reading = make_reading(stack, [(q1, 1, v1)])
        numpy.testing.assert_allclose(get_reading(stack), expected_reading)

        update_stack(stack, [(REPL, q1, 1, q2, 2, 20.0)], v0)
        # Stack: [ (0, 0), (2, v1) ] ; q2 / 10.0
        expected_reading = make_reading(stack, [(q2, 2, v1)])
        numpy.testing.assert_allclose(get_reading(stack), expected_reading)

        update_stack(stack, [(POP, q2, 2, q1, 0.1)], v0)
        # Stack: [ (0, 0) ] ; q1 / 1.0
        expected_reading = make_reading(stack, [])
        numpy.testing.assert_allclose(get_reading(stack), expected_reading)

    def test_spurious_run_has_zero_weight(self):
        device = torch.device('cpu')
        semiring = log
        B = 1
        Q = 3
        S = 4
        m = 3
        v0 = torch.tensor([0.9, 0.1, 0.1], device=device)
        v1 = torch.tensor([0.1, 0.9, 0.1], device=device)
        v2 = torch.tensor([0.1, 0.1, 0.9], device=device)
        stack = get_vector_nondeterministic_stack(
            batch_size=B,
            num_states=Q,
            stack_alphabet_size=S,
            stack_embedding_size=m,
            sequence_length=2,
            bottom_vector=torch.log(v0),
            block_size=32,
            dtype=torch.float32,
            device=device,
            semiring=semiring
        )

        q0, q1, q2 = range(Q)

        # Stack: [ (0, v0) ] ; q0 / 1.0
        expected_reading = make_reading(stack, [(q0, 0, v0)])
        numpy.testing.assert_allclose(get_reading(stack), expected_reading)

        update_stack(stack, [
            (PUSH, q0, 0, q1, 1, 1.0),
            (PUSH, q1, 1, q2, 2, 1.0)
        ], v1)
        # Stack 1: [ (0, v0), (1, v1) ] ; q1 / 1.0
        # Stack 2: [ (0, v0), (2, v1) ] ; q2 / 1.0
        expected_reading = make_reading(stack, [(q1, 1, v1)])
        numpy.testing.assert_allclose(get_reading(stack), expected_reading)

        update_stack(stack, [
            (POP, q1, 1, q2, 1.0),
            (POP, q2, 2, q1, 1.0)
        ], v2)
        # Stack 1: [ (0, v0) ] ; q2 / 1.0
        # Stack 2: [ (0, v0) ] ; q1 / 1.0
        expected_reading = make_reading(stack, [(q2, 0, v0)])
        numpy.testing.assert_allclose(get_reading(stack), expected_reading)

    def test_stack_matches_reference(self):
        device = torch.device('cpu')
        semiring = log
        B = 4
        Q = 2
        S = 3
        m = 5
        n = 7
        generator = torch.manual_seed(123)
        stack = get_vector_nondeterministic_stack(
            batch_size=B,
            num_states=Q,
            stack_alphabet_size=S,
            stack_embedding_size=m,
            sequence_length=n,
            bottom_vector=torch.full((m,), -math.inf),
            block_size=32,
            dtype=torch.float32,
            device=device,
            semiring=semiring
        )
        reference_stack = ReferenceVectorNondeterministicStack(B, Q, S, m, device)
        reading = stack.reading()
        reference_reading = reference_stack.reading()
        numpy.testing.assert_allclose(reading, reference_reading)
        def random_log_tensor(size):
            return torch.empty(size, device=device).uniform_(-0.1, 0.1)
        for t in range(n):
            push = random_log_tensor((B, Q, S, Q, S))
            repl = random_log_tensor((B, Q, S, Q, S))
            pop = random_log_tensor((B, Q, S, Q))
            pushed_vector = torch.nn.functional.logsigmoid(random_log_tensor((B, m)))
            stack.update(push, repl, pop, pushed_vector)
            reading = stack.reading()
            reference_stack.update(push, repl, pop, pushed_vector)
            reference_reading = reference_stack.reading()
            numpy.testing.assert_allclose(reading, reference_reading, rtol=1e-5,
                err_msg=f'does not match reference implementation at timestep t = {t}')

    def test_unlimited_length(self):
        device = torch.device('cpu')
        semiring = log
        B = 4
        Q = 2
        S = 3
        m = 5
        n = 7
        generator = torch.manual_seed(123)
        bottom_vector = torch.full((m,), -math.inf)
        unlimited_stack = get_vector_nondeterministic_stack(
            batch_size=B,
            num_states=Q,
            stack_alphabet_size=S,
            stack_embedding_size=m,
            sequence_length=None,
            bottom_vector=bottom_vector,
            block_size=32,
            dtype=torch.float32,
            device=device,
            semiring=semiring
        )
        limited_stack = get_vector_nondeterministic_stack(
            batch_size=B,
            num_states=Q,
            stack_alphabet_size=S,
            stack_embedding_size=m,
            sequence_length=n,
            bottom_vector=bottom_vector,
            block_size=32,
            dtype=torch.float32,
            device=device,
            semiring=semiring
        )
        def random_log_tensor(size):
            return torch.empty(size, device=device).uniform_(-0.1, 0.1)
        for t in range(n):
            push = random_log_tensor((B, Q, S, Q, S))
            repl = random_log_tensor((B, Q, S, Q, S))
            pop = random_log_tensor((B, Q, S, Q))
            pushed_vector = torch.nn.functional.logsigmoid(random_log_tensor((B, m)))
            unlimited_stack.update(push, repl, pop, pushed_vector)
            limited_stack.update(push, repl, pop, pushed_vector)
            unlimited_reading = unlimited_stack.reading()
            limited_reading = limited_stack.reading()
            numpy.testing.assert_allclose(unlimited_reading, limited_reading,
                err_msg=
                f'reading of unlimited-length stack does not match reading '
                f'of limited-length stack at timestep t = {t}')
        push = random_log_tensor((B, Q, S, Q, S))
        repl = random_log_tensor((B, Q, S, Q, S))
        pop = random_log_tensor((B, Q, S, Q))
        pushed_vector = torch.nn.functional.logsigmoid(random_log_tensor((B, m)))
        for i in range(3):
            unlimited_stack.update(push, repl, pop, pushed_vector)
        with pytest.raises(TooManyUpdates):
            limited_stack.update(push, repl, pop, pushed_vector)

class ReferenceVectorNondeterministicStack(ReferenceNondeterministicStack):

    def __init__(self, B, Q, S, m, device):
        super().__init__(B, Q, S, device)
        self.m = m
        self.zeta = self.semiring.zeros((B, 1, 1, Q, S, Q, S, m), device=device)
        self.zeta[:, self.zeta_i_index(-1), self.zeta_j_index(0), 0, 0, 0, 0] = self.semiring.zero

    def zeta_i_index(self, i):
        return ensure_not_negative(i+1)

    def zeta_j_index(self, j):
        return ensure_not_negative(j)

    def get_zeta(self, b, i, j, q, x, r, y):
        try:
            return self.zeta[b, self.zeta_i_index(i), self.zeta_j_index(j), q, x, r, y]
        except IndexError:
            return self.semiring.zeros((self.m,), device=self.device)

    def update_zeta(self, push, repl, pop, pushed_vector):
        B = self.B
        Q = self.Q
        S = self.S
        m = self.m
        j = self.j
        push = self.convert_operation_weights(push)
        repl = self.convert_operation_weights(repl)
        pop = self.convert_operation_weights(pop)
        pushed_vector = self.convert_operation_weights(pushed_vector)
        zeta_j = recursively_stack([
            [
                [
                    [
                        [
                            [
                                self.compute_zeta(push, repl, pop, pushed_vector, b, i, j, q, x, r, y)
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
        self.zeta = torch.cat([
            torch.cat([
                self.zeta,
                self.semiring.zeros((B, 1, j, Q, S, Q, S, m), device=self.device)
            ], dim=1),
            zeta_j[:, :, None]
        ], dim=2)

    def compute_zeta(self, push, repl, pop, pushed_vector, b, i, j, q, x, r, y):
        Q = self.Q
        S = self.S
        m = self.m
        return self.sum([
            (
                self.semiring.multiply(push[b, q, x, r, y], pushed_vector[b])
                if i == j-1
                else self.semiring.zeros((m,), device=self.device)
            ),
            self.sum([
                self.semiring.multiply(
                    self.get_zeta(b, i, j-1, q, x, s, z),
                    repl[b, s, z, r, y]
                )
                for s in range(Q)
                for z in range(S)
            ]),
            self.sum([
                self.semiring.multiply(
                    self.semiring.multiply(
                        self.get_zeta(b, i, k, q, x, u, y),
                        self.get_gamma(b, k, j-1, u, y, s, z)
                    ),
                    pop[b, s, z, r]
                )
                for k in range(i+1, j-2+1)
                for u in range(Q)
                for s in range(Q)
                for z in range(S)
            ], size=(m,))
        ])

    def update(self, push, repl, pop, pushed_vector):
        super().update(push, repl, pop)
        self.update_zeta(push, repl, pop, pushed_vector)

    def reading(self):
        B = self.B
        Q = self.Q
        S = self.S
        m = self.m
        j = self.j
        eta_j = recursively_stack([
            [
                [
                    self.sum([
                        self.semiring.multiply(
                            self.get_alpha(b, i, q, x),
                            self.get_zeta(b, i, j, q, x, r, y)
                        )
                        for i in range(0, j-1+1)
                        for q in range(Q)
                        for x in range(S)
                    ], size=(m,))
                    for y in range(S)
                ]
                for r in range(Q)
            ]
            for b in range(B)
        ])
        denom = recursively_stack([
            self.sum([
                self.get_alpha(b, j, rr, yy)
                for rr in range(Q)
                for yy in range(S)
            ])
            for b in range(B)
        ])
        r_j = recursively_stack([
            [
                [
                    self.divide_to_real(eta_j[b, r, y], denom[b])
                    for y in range(S)
                ]
                for r in range(Q)
            ]
            for b in range(B)
        ])
        return r_j.view(B, Q * S * m)

if __name__ == '__main__':
    unittest.main()
