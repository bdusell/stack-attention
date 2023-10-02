import unittest

import more_itertools
import numpy
import torch

from torch_unidirectional import UnidirectionalLSTM
from lib.semiring import log
from stack_rnn_models.vector_nondeterministic_stack import (
    VectorNondeterministicStackRNN,
    VectorNondeterministicStack
)
from stack_rnn_models.limited_vector_nondeterministic_stack import (
    LimitedVectorNondeterministicStackRNN,
    LimitedVectorNondeterministicStack
)

class TestLimitedVectorRNSRNN(unittest.TestCase):

    def assert_is_finite(self, tensor, message=None):
        self.assertTrue(torch.all(torch.isfinite(tensor)).item(), message)

    def test_forward_and_backward(self):
        # Test that a simple forward and backward pass for a single chunk does
        # not raise an error and does not produce NaNs.
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        stack_embedding_size = 4
        hidden_units = 11
        sequence_length = 13
        window_size = 5
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = LimitedVectorNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            stack_embedding_size=stack_embedding_size,
            window_size=window_size,
            controller=controller,
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

    def test_truncated_bptt(self):
        # Make sure that the API for iterative execution works as expected,
        # and that the outputs and gradients do not have NaNs.
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        stack_embedding_size = 4
        hidden_units = 11
        sequence_length = 13
        window_size = 5
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = LimitedVectorNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            stack_embedding_size=stack_embedding_size,
            window_size=window_size,
            controller=controller,
            normalize_operations=False
        )
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1, generator=generator)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = torch.nn.MSELoss()
        state = model.initial_state(batch_size)

        # First iteration
        optimizer.zero_grad()
        self.assertEqual(state.batch_size(), batch_size)
        input_tensor = torch.empty(batch_size, sequence_length, input_size)
        input_tensor.uniform_(-1, 1, generator=generator)
        predicted_tensor, state = model(
            input_tensor,
            initial_state=state,
            return_state=True,
            include_first=False,
            block_size=10
        )
        state = state.detach()
        self.assertEqual(
            predicted_tensor.size(),
            (batch_size, sequence_length, hidden_units),
            'output does not have expected dimensions'
        )
        self.assert_is_finite(predicted_tensor)
        target_tensor = torch.empty(batch_size, sequence_length, hidden_units)
        target_tensor.uniform_(-1, 1, generator=generator)
        loss = criterion(predicted_tensor, target_tensor)
        loss.backward()
        for name, p in model.named_parameters():
            self.assert_is_finite(p.grad, f'gradient for parameter {name} is not finite')
        optimizer.step()

        # Second iteration with smaller batch size.
        self.assertEqual(state.batch_size(), batch_size)
        batch_size -= 1
        state = state.slice_batch(slice(batch_size))
        self.assertEqual(state.batch_size(), batch_size)
        optimizer.zero_grad()
        input_tensor = torch.empty(batch_size, sequence_length, input_size)
        input_tensor.uniform_(-1, 1, generator=generator)
        predicted_tensor, state = model(
            input_tensor,
            initial_state=state,
            return_state=True,
            include_first=False,
            block_size=10
        )
        state = state.detach()
        self.assertEqual(
            predicted_tensor.size(),
            (batch_size, sequence_length, hidden_units),
            'output does not have expected dimensions'
        )
        self.assert_is_finite(predicted_tensor)
        target_tensor = torch.empty(batch_size, sequence_length, hidden_units)
        target_tensor.uniform_(-1, 1, generator=generator)
        loss = criterion(predicted_tensor, target_tensor)
        loss.backward()
        for name, p in model.named_parameters():
            self.assert_is_finite(p.grad, f'gradient for parameter {name} is not finite')
        optimizer.step()

    def test_limited_stack_matches_unlimited_stack(self):
        # Test that the limited vector stack when run on a single input chunk
        # with a window size (D) as large as the input sequence length matches
        # the output of the unlimited vector stack.
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        stack_embedding_size = 4
        hidden_units = 11
        sequence_length = 13
        generator = torch.manual_seed(123)
        semiring = log
        device = torch.device('cpu')
        def random_tensor(size, lo, hi):
            return torch.empty(size).uniform_(lo, hi, generator=generator)
        bottom_vector = torch.log(random_tensor((stack_embedding_size,), 0.01, 0.99))
        limited_stack = LimitedVectorNondeterministicStack(
            batch_size=batch_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            stack_embedding_size=stack_embedding_size,
            sequence_length=sequence_length,
            window_size=sequence_length,
            bottom_vector=bottom_vector,
            initial_state=None,
            semiring=semiring,
            block_size=10,
            dtype=torch.float32,
            device=device
        )
        unlimited_stack = VectorNondeterministicStack(
            batch_size=batch_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            stack_embedding_size=stack_embedding_size,
            sequence_length=sequence_length,
            bottom_vector=bottom_vector,
            block_size=10,
            dtype=torch.float32,
            device=device,
            semiring=semiring
        )
        B = batch_size
        Q = num_states
        S = stack_alphabet_size
        m = stack_embedding_size
        limited_reading = limited_stack.reading()
        unlimited_reading = unlimited_stack.reading()
        numpy.testing.assert_allclose(
            limited_reading.detach(),
            unlimited_reading.detach(),
            err_msg=f'stack reading of limited vector nondeterministic '
                    f'stack does not match unlimited version at timestep 0')
        for i in range(1, sequence_length):
            push = torch.log(random_tensor((B, Q, S, Q, S), 0.01, 10))
            repl = torch.log(random_tensor((B, Q, S, Q, S), 0.01, 10))
            pop = torch.log(random_tensor((B, Q, S, Q), 0.01, 10))
            pushed_vector = torch.log(random_tensor((B, m), 0.01, 0.99))
            limited_stack.update(push, repl, pop, pushed_vector)
            unlimited_stack.update(push, repl, pop, pushed_vector)
            limited_reading = limited_stack.reading()
            unlimited_reading = unlimited_stack.reading()
            numpy.testing.assert_allclose(
                limited_reading.detach(),
                unlimited_reading.detach(),
                rtol=1e-5,
                err_msg=f'stack reading of limited vector nondeterministic '
                        f'stack does not match unlimited version at timestep '
                        f'{i}')

    def test_limited_rnn_matches_unlimited_rnn(self):
        # Test that limited vector RNS-RNN when run on a single input chunk
        # with a window size (D) as large as the input length matches the
        # unlimited vector RNS-RNN.
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
        limited_model = LimitedVectorNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            stack_embedding_size=stack_embedding_size,
            window_size=sequence_length,
            controller=controller,
            normalize_operations=False
        )
        for p in limited_model.parameters():
            p.data.uniform_(-0.1, 0.1, generator=generator)
        unlimited_model = VectorNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            stack_embedding_size=stack_embedding_size,
            controller=controller,
            normalize_operations=False
        )
        for unlimited_p, limited_p in more_itertools.zip_equal(unlimited_model.parameters(), limited_model.parameters()):
            unlimited_p.data.copy_(limited_p.data)
        input_tensor = torch.empty(batch_size, sequence_length - 1, input_size)
        input_tensor.uniform_(-1, 1, generator=generator)
        limited_predicted_tensor = limited_model(input_tensor, block_size=10)
        unlimited_predicted_tensor = unlimited_model(input_tensor, block_size=10)
        numpy.testing.assert_allclose(
            limited_predicted_tensor.detach(),
            unlimited_predicted_tensor.detach(),
            rtol=1e-4,
            err_msg='output of limited vector RNS-RNN does not match output '
                    'of unlimited vector RNS-RNN')
        criterion = torch.nn.MSELoss()
        target_tensor = torch.empty(batch_size, sequence_length, hidden_units)
        target_tensor.uniform_(-1, 1, generator=generator)
        limited_loss = criterion(limited_predicted_tensor, target_tensor)
        limited_loss.backward()
        unlimited_loss = criterion(unlimited_predicted_tensor, target_tensor)
        unlimited_loss.backward()
        for limited_p, unlimited_p in more_itertools.zip_equal(limited_model.parameters(), unlimited_model.parameters()):
            numpy.testing.assert_allclose(
                limited_p.grad,
                unlimited_p.grad,
                rtol=1e-3,
                err_msg='gradient of limited vector RNS-RNN does not match '
                        'gradient of unlimited vector RNS-RNN')

    def test_incremental_limited_rnn_matches_unlimited_rnn(self):
        # Test that processing a sequence in chunks with the limited vector
        # RNS-RNN when the window size (D) is as long as the input produces the
        # same result as processing it with the unlimited vector RNS-RNN.
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        stack_embedding_size = 4
        hidden_units = 11
        chunk_lengths = [7, 7, 7]
        sequence_length = sum(chunk_lengths)
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        limited_model = LimitedVectorNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            stack_embedding_size=stack_embedding_size,
            window_size=sequence_length,
            controller=controller,
            normalize_operations=False
        )
        for p in limited_model.parameters():
            p.data.uniform_(-0.1, 0.1, generator=generator)
        unlimited_model = VectorNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            stack_embedding_size=stack_embedding_size,
            controller=controller,
            normalize_operations=False
        )
        for unlimited_p, limited_p in more_itertools.zip_equal(unlimited_model.parameters(), limited_model.parameters()):
            unlimited_p.data.copy_(limited_p.data)

        input_tensor = torch.empty(batch_size, sequence_length, input_size)
        input_tensor.uniform_(-1, 1, generator=generator)

        expected_output = unlimited_model(
            input_tensor,
            include_first=False,
            block_size=10
        )

        predicted_outputs = []
        offset = 0
        state = limited_model.initial_state(batch_size)
        for chunk_length in chunk_lengths:
            predicted_output, state = limited_model(
                input_tensor[:, offset:offset+chunk_length],
                initial_state=state,
                return_state=True,
                include_first=False,
                block_size=10
            )
            predicted_outputs.append(predicted_output)
            state = state.detach()
            offset += chunk_length
        predicted_output = torch.cat(predicted_outputs, dim=1)
        self.assertEqual(predicted_output.size(), expected_output.size())
        for i in range(predicted_output.size(1)):
            # Later timesteps seem to accumulate more and more error, but this
            # shouldn't be a big deal.
            numpy.testing.assert_allclose(
                predicted_output[:, i].detach(),
                expected_output[:, i].detach(),
                rtol=1e-2,
                err_msg=f'mismatch at timestep {i+1}')

if __name__ == '__main__':
    unittest.main()
