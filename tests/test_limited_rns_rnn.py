import itertools
import math
import unittest

import numpy
import torch

from torch_unidirectional import UnidirectionalLSTM
from stack_rnn_models.nondeterministic_stack import NondeterministicStackRNN
from stack_rnn_models.limited_nondeterministic_stack import (
    LimitedNondeterministicStackRNN,
    LimitedNondeterministicStack,
    Operation,
    PushOperation,
    ReplaceOperation,
    PopOperation
)
from lib.semiring import log
from reference_util import recursively_stack
from test_rns_rnn import (
    ReferenceNondeterministicStackRNN,
    ReferenceNondeterministicStack,
    LogSemiring,
    make_reading,
    get_reading,
    update_stack,
    PUSH,
    REPL,
    POP
)

class TestLimitedNondeterministicStackRNN(unittest.TestCase):

    def assert_is_finite(self, tensor, message=None):
        self.assertTrue(torch.all(torch.isfinite(tensor)).item(), message)

    def test_forward_and_backward(self):
        # Test that a simple forward and backward pass for a single chunk does
        # not raise an error and does not produce NaNs.
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        hidden_units = 11
        sequence_length = 13
        window_size = 5
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = LimitedNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            window_size=window_size,
            controller=controller,
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

    def test_stack_nan(self):
        # Test that the stack data structure does not cause NaN gradients.
        batch_size = 5
        num_states = 2
        stack_alphabet_size = 3
        sequence_length = 2
        window_size = 5
        generator = torch.manual_seed(123)
        stack = LimitedNondeterministicStack(
            batch_size=batch_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            sequence_length=sequence_length,
            window_size=window_size,
            include_states_in_reading=True,
            initial_state=None,
            semiring=log,
            block_size=10,
            dtype=torch.float32,
            device=torch.device('cpu')
        )
        B = batch_size
        Q = num_states
        S = stack_alphabet_size
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.push = torch.nn.Parameter(torch.empty(B, Q, S, Q, S))
                self.repl = torch.nn.Parameter(torch.empty(B, Q, S, Q, S))
                self.pop = torch.nn.Parameter(torch.empty(B, Q, S, Q))
        model = Model()
        for p in model.parameters():
            p.data.uniform_(-10, 10, generator=generator)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        terms = []
        terms.append(stack.reading())
        for i in range(1, sequence_length):
            stack.update(model.push, model.repl, model.pop)
            terms.append(stack.reading())
        for term in terms:
            self.assert_is_finite(term)
        loss = torch.sum(torch.stack(terms))
        self.assert_is_finite(loss)
        optimizer.zero_grad()
        loss.backward()
        for name, p in model.named_parameters():
            self.assert_is_finite(p.grad, f'gradient for parameter {name} is not finite')
        optimizer.step()

    def test_forward_and_backward_against_reference(self):
        # Test that the output and gradients of a single chunk match the slow
        # reference implementation.
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        hidden_units = 11
        sequence_length = 13
        window_size = 5
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = LimitedNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            window_size=window_size,
            controller=controller,
            normalize_operations=False,
            include_states_in_reading=True
        )
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1, generator=generator)
        reference_model = ReferenceLimitedNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            window_size=window_size,
            controller=controller,
            normalize_operations=False,
            include_states_in_reading=True
        )
        for p, reference_p in itertools.zip_longest(model.parameters(), reference_model.parameters()):
            reference_p.data.copy_(p.data)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = torch.nn.MSELoss()
        input_tensor = torch.empty(batch_size, sequence_length - 1, input_size)
        input_tensor.uniform_(-1, 1, generator=generator)
        predicted_tensor = model(input_tensor, block_size=10)
        reference_predicted_tensor = reference_model(input_tensor)
        numpy.testing.assert_allclose(
            predicted_tensor.detach(),
            reference_predicted_tensor.detach(),
            rtol=1e-4,
            err_msg='output of forward() does not match reference implementation')
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
                err_msg='gradient does not match reference implementation')

    def test_truncated_bptt(self):
        # Make sure that the API for iterative execution works as expected,
        # and that the outputs and gradients do not have NaNs.
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        hidden_units = 11
        sequence_length = 13
        window_size = 5
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = LimitedNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            window_size=window_size,
            controller=controller,
            normalize_operations=False,
            include_states_in_reading=True
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

    def test_reference_limited_matches_reference_unlimited(self):
        # Test that the forward and backward pass of a single chunk of the
        # reference implementation of the RNS-RNN matches the reference
        # implementation of the unlimited RNS-RNN when the window size is as
        # big as the whole sequence. These should be mathematically equivalent.
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        hidden_units = 11
        sequence_length = 6
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        limited_model = ReferenceLimitedNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            window_size=sequence_length,
            controller=controller,
            normalize_operations=False,
            include_states_in_reading=True
        )
        for p in limited_model.parameters():
            p.data.uniform_(-0.1, 0.1, generator=generator)
        unlimited_model = ReferenceNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            controller=controller,
            normalize_operations=False,
            include_states_in_reading=True
        )
        for unlimited_p, limited_p in itertools.zip_longest(unlimited_model.parameters(), limited_model.parameters()):
            unlimited_p.data.copy_(limited_p.data)
        input_tensor = torch.empty(batch_size, sequence_length - 1, input_size)
        input_tensor.uniform_(-1, 1, generator=generator)
        limited_predicted_tensor = limited_model(input_tensor)
        unlimited_predicted_tensor = unlimited_model(input_tensor)
        numpy.testing.assert_allclose(
            limited_predicted_tensor.detach(),
            unlimited_predicted_tensor.detach(),
            err_msg='output of limited reference implementation does not '
                    'match output of unlimited reference implementation')
        criterion = torch.nn.MSELoss()
        target_tensor = torch.empty(batch_size, sequence_length, hidden_units)
        target_tensor.uniform_(-1, 1, generator=generator)
        limited_loss = criterion(limited_predicted_tensor, target_tensor)
        limited_loss.backward()
        unlimited_loss = criterion(unlimited_predicted_tensor, target_tensor)
        unlimited_loss.backward()
        for limited_p, unlimited_p in itertools.zip_longest(limited_model.parameters(), unlimited_model.parameters()):
            numpy.testing.assert_allclose(
                limited_p.grad,
                unlimited_p.grad,
                err_msg='gradient of limited reference implementation does '
                        'not match gradient of unlimited reference '
                        'implementation')

    def test_large_window_size(self):
        # Test that the limited RNS-RNN works when the window size (D) is
        # larger than the sequence length. Verify this by checking the output
        # and gradients against the unlimited RNS-RNN for a single chunk.
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        hidden_units = 11
        sequence_length = 6
        window_size = 20
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        limited_model = LimitedNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            window_size=window_size,
            controller=controller,
            normalize_operations=False,
            include_states_in_reading=True
        )
        for p in limited_model.parameters():
            p.data.uniform_(-0.1, 0.1, generator=generator)
        unlimited_model = NondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            controller=controller,
            normalize_operations=False,
            include_states_in_reading=True
        )
        for unlimited_p, limited_p in itertools.zip_longest(unlimited_model.parameters(), limited_model.parameters()):
            unlimited_p.data.copy_(limited_p.data)
        input_tensor = torch.empty(batch_size, sequence_length - 1, input_size)
        input_tensor.uniform_(-1, 1, generator=generator)
        limited_predicted_tensor = limited_model(input_tensor, block_size=10)
        unlimited_predicted_tensor = unlimited_model(input_tensor, block_size=10)
        numpy.testing.assert_allclose(
            limited_predicted_tensor.detach(),
            unlimited_predicted_tensor.detach(),
            rtol=1e-4,
            err_msg='output of limited reference implementation does not '
                    'match output of unlimited reference implementation')
        criterion = torch.nn.MSELoss()
        target_tensor = torch.empty(batch_size, sequence_length, hidden_units)
        target_tensor.uniform_(-1, 1, generator=generator)
        limited_loss = criterion(limited_predicted_tensor, target_tensor)
        limited_loss.backward()
        unlimited_loss = criterion(unlimited_predicted_tensor, target_tensor)
        unlimited_loss.backward()
        for limited_p, unlimited_p in itertools.zip_longest(limited_model.parameters(), unlimited_model.parameters()):
            numpy.testing.assert_allclose(
                limited_p.grad,
                unlimited_p.grad,
                rtol=1e-3,
                err_msg='gradient of limited reference implementation does '
                        'not match gradient of unlimited reference '
                        'implementation')

    def test_chunks_match_whole(self):
        # Check that processing a sequence in chunks with the limited RNS-RNN
        # produces the same mathematical result as processing it in one big
        # chunk with the limited RNS-RNN. Here we only check the output; the
        # gradients are expected to be different due to truncated BPTT.
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        hidden_units = 11
        sequence_length = 13
        num_chunks = 3
        window_size = 5
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = LimitedNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            window_size=window_size,
            controller=controller,
            normalize_operations=False,
            include_states_in_reading=True
        )
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1, generator=generator)

        input_tensor = torch.empty(batch_size, sequence_length * num_chunks, input_size)
        input_tensor.uniform_(-1, 1, generator=generator)

        # Process the whole sequence at once and use this as the expected result.
        expected_output, _ = model(
            input_tensor,
            initial_state=model.initial_state(batch_size),
            return_state=True,
            include_first=False,
            block_size=10
        )

        # Process the same sequence in chunks and concatenate them together.
        predicted_outputs = []
        state = model.initial_state(batch_size)
        for chunk_no in range(num_chunks):
            offset = chunk_no * sequence_length
            predicted_output, state = model(
                input_tensor[:, offset:offset+sequence_length],
                initial_state=state,
                return_state=True,
                include_first=False,
                block_size=10
            )
            predicted_outputs.append(predicted_output)
            state = state.detach()
        predicted_output = torch.cat(predicted_outputs, dim=1)

        numpy.testing.assert_allclose(
            predicted_output.detach(),
            expected_output.detach(),
            err_msg='output of chunks does not match output of whole sequence')

    def test_chunks_match_unlimited(self):
        # Check that processing a sequence in chunks with the limited RNS-RNN
        # when the window size (D) is as long as the sequence length produces
        # the same result as processing it with the unlimited RNS-RNN. Here we
        # only check the output; the gradients are expected to be different due
        # to truncated BPTT.
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        hidden_units = 11
        sequence_length = 13
        num_chunks = 3
        window_size = num_chunks * sequence_length
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        limited_model = LimitedNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            window_size=window_size,
            controller=controller,
            normalize_operations=False,
            include_states_in_reading=True
        )
        for p in limited_model.parameters():
            p.data.uniform_(-0.1, 0.1, generator=generator)
        unlimited_model = NondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            controller=controller,
            normalize_operations=False,
            include_states_in_reading=True
        )
        for unlimited_p, limited_p in itertools.zip_longest(unlimited_model.parameters(), limited_model.parameters()):
            unlimited_p.data.copy_(limited_p.data)

        input_tensor = torch.empty(batch_size, sequence_length * num_chunks, input_size)
        input_tensor.uniform_(-1, 1, generator=generator)

        # Process the whole sequence using the unlimited model and use this as
        # the expected result.
        expected_output = unlimited_model(
            input_tensor,
            include_first=False,
            block_size=10
        )

        # Process the same sequence in chunks and concatenate them together.
        predicted_outputs = []
        state = limited_model.initial_state(batch_size)
        for chunk_no in range(num_chunks):
            offset = chunk_no * sequence_length
            predicted_output, state = limited_model(
                input_tensor[:, offset:offset+sequence_length],
                initial_state=state,
                return_state=True,
                include_first=False,
                block_size=10
            )
            predicted_outputs.append(predicted_output)
            state = state.detach()
        predicted_output = torch.cat(predicted_outputs, dim=1)

        numpy.testing.assert_allclose(
            predicted_output.detach(),
            expected_output.detach(),
            rtol=1e-4,
            err_msg='output of chunks does not match output of whole sequence')

    def test_large_window_size_chunks(self):
        # Check that processing a sequence in chunks with the limited RNS-RNN
        # still works correctly when the window size (D) is longer than the
        # chunk length.
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        hidden_units = 11
        chunk_lengths = [3, 4, 10, 5]
        window_size = 8
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = LimitedNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            window_size=window_size,
            controller=controller,
            normalize_operations=False,
            include_states_in_reading=True
        )
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1, generator=generator)

        input_tensor = torch.empty(batch_size, sum(chunk_lengths), input_size)
        input_tensor.uniform_(-1, 1, generator=generator)

        # Process the whole sequence at once and use this as the expected result.
        expected_output, _ = model(
            input_tensor,
            initial_state=model.initial_state(batch_size),
            return_state=True,
            include_first=False,
            block_size=10
        )

        # Process the same sequence in chunks and concatenate them together.
        predicted_outputs = []
        state = model.initial_state(batch_size)
        for chunk in input_tensor.split(chunk_lengths, dim=1):
            predicted_output, state = model(
                chunk,
                initial_state=state,
                return_state=True,
                include_first=False,
                block_size=10
            )
            predicted_outputs.append(predicted_output)
            state = state.detach()
        predicted_output = torch.cat(predicted_outputs, dim=1)

        numpy.testing.assert_allclose(
            predicted_output.detach(),
            expected_output.detach(),
            err_msg='output of chunks does not match output of whole sequence')

    def test_return_actions(self):
        # Test that the model can return the operation weights at every time
        # step.
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        hidden_units = 11
        sequence_length = 13
        window_size = 5
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = LimitedNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            window_size=window_size,
            controller=controller,
            normalize_operations=False,
            include_states_in_reading=True
        )
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1, generator=generator)
        input_tensor = torch.empty(batch_size, sequence_length, input_size)
        input_tensor.uniform_(-1, 1, generator=generator)
        B = batch_size
        T = sequence_length
        H = hidden_units
        Q = num_states
        S = stack_alphabet_size
        model.eval()
        with torch.no_grad():

            output, actions = model(
                input_tensor,
                include_first=False,
                return_actions=True,
                block_size=10
            )
            self.assertEqual(output.size(), (B, T, H))
            self.assertEqual(len(actions), T)
            self.assertIsNone(actions[0])
            for action in actions[1:]:
                push, repl, pop = action
                self.assertEqual(push.size(), (B, Q, S, Q, S))
                self.assertEqual(repl.size(), (B, Q, S, Q, S))
                self.assertEqual(pop.size(), (B, Q, S, Q))

            (output, actions), state = model(
                input_tensor,
                include_first=False,
                return_actions=True,
                return_state=True,
                block_size=10
            )
            self.assertEqual(output.size(), (B, T, H))
            self.assertEqual(len(actions), T)
            self.assertIsNone(actions[0])
            for action in actions[1:]:
                push, repl, pop = action
                self.assertEqual(push.size(), (B, Q, S, Q, S))
                self.assertEqual(repl.size(), (B, Q, S, Q, S))
                self.assertEqual(pop.size(), (B, Q, S, Q))
            self.assertIsInstance(state, LimitedNondeterministicStackRNN.InitialState)

    def test_viterbi(self):
        # Test that the Viterbi algorithm runs without error.
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        hidden_units = 11
        sequence_length = 13
        window_size = 5
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = LimitedNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            window_size=window_size,
            controller=controller,
            normalize_operations=False,
            include_states_in_reading=True
        )
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1, generator=generator)
        input_tensor = torch.empty(batch_size, sequence_length, input_size)
        input_tensor.uniform_(-1, 1, generator=generator)
        model.eval()
        decoder = model.viterbi_decoder(input_tensor, block_size=10)
        paths, scores = decoder.decode_timestep(sequence_length-1)
        for path in paths:
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
        window_size = 5
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = LimitedNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            window_size=window_size,
            controller=controller,
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

    def test_push_repl_pop(self):
        device = torch.device('cpu')
        semiring = log
        B = 1
        Q = 3
        S = 4
        n = 4
        stack = LimitedNondeterministicStack(
            batch_size=B,
            num_states=Q,
            stack_alphabet_size=S,
            sequence_length=n,
            window_size=n,
            include_states_in_reading=True,
            initial_state=None,
            semiring=semiring,
            block_size=32,
            dtype=torch.float32,
            device=device
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

class ReferenceLimitedNondeterministicStackRNN(NondeterministicStackRNN):

    def __init__(self, *args, window_size, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = window_size

    def initial_stack(self, batch_size, stack_size, sequence_length):
        tensor = next(self.parameters())
        return ReferenceLimitedNondeterministicStack(
            batch_size,
            self.num_states,
            self.stack_alphabet_size,
            self.window_size,
            tensor.device
        )

class ReferenceLimitedNondeterministicStack(ReferenceNondeterministicStack):

    def __init__(self, B, Q, S, D, device):
        super().__init__(B, Q, S, device)
        self.D = D

    def compute_gamma(self, push, repl, pop, b, i, j, q, x, r, y):
        D = self.D
        if i >= j-D:
            return super().compute_gamma(push, repl, pop, b, i, j, q, x, r, y)
        else:
            return self.semiring.zeros((), device=self.device)

if __name__ == '__main__':
    unittest.main()
