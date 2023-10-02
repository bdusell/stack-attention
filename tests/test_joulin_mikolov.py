import unittest

import numpy
import torch

from torch_unidirectional import UnidirectionalLSTM
from stack_rnn_models.joulin_mikolov import JoulinMikolovRNN

class TestJoulinMikolovRNN(unittest.TestCase):

    def test_joulin_mikolov(self):
        stack_embedding_size = 3
        input_size = 5
        hidden_units = 7
        batch_size = 11
        sequence_length = 13
        generator = torch.manual_seed(0)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = JoulinMikolovRNN(
            input_size=input_size,
            stack_embedding_size=stack_embedding_size,
            controller=controller
        )
        for p in model.parameters():
            p.data.uniform_(generator=generator)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        input_tensor = torch.empty(batch_size, sequence_length - 1, input_size)
        input_tensor.uniform_(generator=generator)
        predicted_tensor = model(input_tensor)
        self.assertEqual(
            predicted_tensor.size(),
            (batch_size, sequence_length, hidden_units),
            'output has the expected dimensions'
        )
        target_tensor = torch.empty(batch_size, sequence_length, hidden_units)
        target_tensor.uniform_(generator=generator)
        loss = criterion(predicted_tensor, target_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def test_even_length_hint(self):
        self._test_length_hint(20)

    def test_odd_length_hint(self):
        self._test_length_hint(21)

    def _test_length_hint(self, sequence_length):
        stack_embedding_size = 3
        input_size = 5
        hidden_units = 7
        batch_size = 11
        device = torch.device('cpu')
        generator = torch.manual_seed(0)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = JoulinMikolovRNN(
            input_size=input_size,
            stack_embedding_size=stack_embedding_size,
            controller=controller
        )
        for p in model.parameters():
            p.data.uniform_(generator=generator)
        input_tensor = torch.empty(batch_size, sequence_length, input_size)
        input_tensor.uniform_(generator=generator)
        predicted_tensor_with_length = model(input_tensor)
        self.assertEqual(
            predicted_tensor_with_length.size(),
            (batch_size, sequence_length + 1, hidden_units))
        state_no_length = model.initial_state(batch_size)
        predicted_tensor_no_length = state_no_length.forward(
            input_tensor, return_state=False, include_first=True)
        self.assertEqual(
            predicted_tensor_no_length.size(),
            (batch_size, sequence_length + 1, hidden_units))
        numpy.testing.assert_allclose(
            predicted_tensor_with_length.detach(),
            predicted_tensor_no_length.detach())
        predicted_tensor_reference = model(
            input_tensor,
            stack_constructor=construct_reference_stack
        )
        numpy.testing.assert_allclose(
            predicted_tensor_with_length.detach(),
            predicted_tensor_reference.detach())

    def test_against_reference(self):
        stack_embedding_size = 3
        input_size = 5
        hidden_units = 7
        batch_size = 11
        sequence_length = 20
        generator = torch.manual_seed(0)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = JoulinMikolovRNN(
            input_size=input_size,
            stack_embedding_size=stack_embedding_size,
            controller=controller
        )
        for name, p in model.named_parameters():
            p.data.uniform_(generator=generator)
        input_tensor = torch.empty(batch_size, sequence_length, input_size)
        input_tensor.uniform_(generator=generator)
        predicted_tensor = model(input_tensor)
        predicted_tensor_reference = model(
            input_tensor,
            stack_constructor=construct_reference_stack
        )
        numpy.testing.assert_allclose(
            predicted_tensor.detach(),
            predicted_tensor_reference.detach())

    def test_return_actions(self):
        stack_embedding_size = 3
        input_size = 5
        hidden_units = 7
        batch_size = 11
        input_length = 8
        generator = torch.manual_seed(0)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = JoulinMikolovRNN(
            input_size=input_size,
            stack_embedding_size=stack_embedding_size,
            controller=controller
        )
        for name, p in model.named_parameters():
            p.data.uniform_(generator=generator)
        input_tensor = torch.empty(batch_size, input_length, input_size)
        input_tensor.uniform_(generator=generator)
        predicted_tensor, actions = model(input_tensor, return_actions=True)
        self.assertIsInstance(predicted_tensor, torch.Tensor)
        self.assertEqual(
            predicted_tensor.size(),
            (batch_size, input_length + 1, hidden_units),
            'output has the expected dimensions'
        )
        self.assertIsInstance(actions, list)
        self.assertEqual(len(actions), input_length + 1)
        # The actions returned at each timestep represent the actions emitted
        # after the previous timestep to compute the current timestep.
        # The actions for the first two timesteps should be None. For timestep
        # 0, it is None because the initial state has no previous state and no
        # previous actions. For timestep 1, it is None because the stack is
        # initialized from scratch just before that timstep, so again there are
        # no previous stack actions.
        for action in actions[:2]:
            self.assertIsNone(action)
        for action in actions[2:]:
            self.assertIsInstance(action, torch.Tensor)
            self.assertEqual(action.size(), (batch_size, 1, 3))

    def test_truncated_bptt(self):
        batch_size = 5
        input_size = 7
        hidden_units = 11
        stack_embedding_size = hidden_units
        sequence_length = 13
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = JoulinMikolovRNN(
            input_size=input_size,
            stack_embedding_size=stack_embedding_size,
            controller=controller,
            push_hidden_state=True,
            stack_depth_limit=10
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
            include_first=False
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
            include_first=False
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

    def assert_is_finite(self, tensor, message=None):
        self.assertTrue(torch.all(torch.isfinite(tensor)).item(), message)

class ReferenceJoulinMikolovStack:

    def __init__(self, elements, bottom):
        self.elements = elements
        self.bottom = bottom

    def reading(self):
        if self.elements:
            return self.elements[0]
        else:
            return self.bottom

    def next(self, push_prob, pop_prob, noop_prob, push_value):
        return ReferenceJoulinMikolovStack(
            list(self.next_elements(push_prob, pop_prob, noop_prob, push_value)),
            self.bottom
        )

    def next_elements(self, push_prob, pop_prob, noop_prob, push_value):
        new_stack_height = len(self.elements) + 1
        for i in range(new_stack_height):
            if i > 0:
                above = self.elements[i-1]
            else:
                above = push_value
            if i < new_stack_height - 1:
                same = self.elements[i]
            else:
                same = self.bottom
            if i < new_stack_height - 2:
                below = self.elements[i+1]
            else:
                below = self.bottom
            yield push_prob[:, :] * above + noop_prob[:, :] * same + pop_prob[:, :] * below

def construct_reference_stack(batch_size, stack_size, sequence_length, max_depth, device):
    return ReferenceJoulinMikolovStack(
        [],
        torch.zeros(batch_size, stack_size, device=device)
    )

if __name__ == '__main__':
    unittest.main()
