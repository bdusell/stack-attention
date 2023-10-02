import typing

import torch

from torch_extras.layer import Layer
from .common import StackRNNBase

class GrefenstetteRNN(StackRNNBase):

    def __init__(self,
            input_size: int,
            stack_embedding_size: int,
            controller: typing.Callable,
            controller_output_size: int
        ):
        super().__init__(
            input_size=input_size,
            stack_reading_size=stack_embedding_size,
            controller=controller,
            controller_output_size=controller_output_size
        )
        self.action_layer = Layer(
            controller_output_size,
            2,
            torch.nn.Sigmoid()
        )
        self.push_value_layer = Layer(
            controller_output_size,
            stack_embedding_size,
            torch.nn.Tanh()
        )

    def initial_stack(self, batch_size, reading_size, sequence_length=None):
        return GrefenstetteStack(
            elements=[],
            bottom=torch.zeros((batch_size, reading_size), device=next(self.parameters()).device)
        )

    class State(StackRNNBase.State):

        def compute_stack(self, hidden_state, stack):
            actions = self.rnn.action_layer(hidden_state)
            push_value = self.rnn.push_value_layer(hidden_state)
            return stack.next(actions, push_value), actions

class GrefenstetteStack:

    def __init__(self, elements, bottom):
        self.elements = elements
        self.bottom = bottom

    def reading(self):
        device = self.bottom.device
        batch_size = self.bottom.size(0)
        result = self.bottom
        strength_left = torch.ones((batch_size,), device=device)
        for value, strength in reversed(self.elements):
            result = result + value * torch.min(
                strength,
                torch.nn.functional.relu(strength_left)
            )[:, None]
            strength_left = strength_left - strength
        return result

    def next(self, actions, push_value):
        return GrefenstetteStack(
            self.next_elements(actions, push_value),
            self.bottom
        )

    def next_elements(self, actions, push_value):
        push_strength = actions[:, 0]
        pop_strength = actions[:, 1]
        result = []
        strength_left = pop_strength
        for value, strength in reversed(self.elements):
            result.append((
                value,
                torch.nn.functional.relu(
                    strength -
                    torch.nn.functional.relu(strength_left)
                )
            ))
            strength_left = strength_left - strength
        result.reverse()
        result.append((push_value, push_strength))
        return result
