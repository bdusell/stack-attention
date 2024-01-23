from collections.abc import Callable, Iterable
import dataclasses
from typing import Any, Union

import more_itertools
import torch

from stack_rnn_models.stack import DifferentiableStack
from torch_unidirectional import Unidirectional, ForwardResult
from torch_extras.layer import Layer

class StackAttention(Unidirectional):

    def __init__(self,
        d_model: int,
        num_actions: int,
        pushed_vector_size: int,
        stack_reading_size: int
    ):
        super().__init__()
        self.action_layer = Layer(d_model, num_actions, bias=False)
        self.pushed_vector_size = pushed_vector_size
        self.input_to_pushed_vector_layer = Layer(d_model, pushed_vector_size, bias=False)
        self.stack_reading_size = stack_reading_size
        self.stack_reading_to_output_layer = Layer(stack_reading_size, d_model, bias=False)

    @dataclasses.dataclass
    class State(Unidirectional.State):

        parent: 'StackAttention'
        stack: DifferentiableStack
        return_actions: bool

        def next(self, input_tensor: torch.Tensor) -> Unidirectional.State:
            # TODO It should be possible to compute all the actions and pushed
            # vectors in parallel, but this would require a complicated
            # refactoring to allow Unidirectionals to have multiple inputs and
            # outputs. And if the whole model is being decoded incrementally,
            # it doesn't matter anyway.
            return dataclasses.replace(
                self,
                stack=self.parent.next_stack(
                    self.stack,
                    self.parent.transform_actions(self.parent.action_layer(input_tensor)),
                    self.parent.input_to_pushed_vector_layer(input_tensor)
                )
            )

        def output(self) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
            return self.parent.stack_reading_to_output_layer(self.stack.reading())

        def batch_size(self) -> int:
            return self.stack.batch_size()

        def transform_tensors(self,
            func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Unidirectional.State:
            return dataclasses.replace(
                self,
                stack=self.stack.transform_tensors(func)
            )

        def _outputs_and_stack(self,
            input_sequence: torch.Tensor,
            include_first: bool
        ):
            action_sequence = self.parent.transform_actions(self.parent.action_layer(input_sequence))
            pushed_vector_sequence = self.parent.input_to_pushed_vector_layer(input_sequence)
            stack = self.stack
            reading_list = []
            if include_first:
                reading_list.append(stack.reading())
            for action_tensor, pushed_vector in zip(
                action_sequence.transpose(0, 1),
                pushed_vector_sequence.transpose(0, 1)
            ):
                stack = self.parent.next_stack(stack, action_tensor, pushed_vector)
                reading_list.append(stack.reading())
            reading_sequence = torch.stack(reading_list, dim=1)
            outputs = self.parent.stack_reading_to_output_layer(reading_sequence)
            extra_outputs = []
            if self.return_actions:
                extra_outputs.append(action_sequence.transpose(0, 1))
            return outputs, stack, extra_outputs

        def outputs(self,
            input_sequence: torch.Tensor,
            include_first: bool
        ) -> Union[Iterable[torch.Tensor], Iterable[tuple[torch.Tensor, ...]]]:
            outputs, stack, extra_outputs = self._outputs_and_stack(input_sequence, include_first)
            if extra_outputs:
                outputs = more_itertools.zip_equal(outputs, *extra_outputs)
            return outputs

        def forward(self,
            input_sequence: torch.Tensor,
            return_state: bool,
            include_first: bool
        ) -> Union[torch.Tensor, ForwardResult]:
            outputs, stack, extra_outputs = self._outputs_and_stack(input_sequence, include_first)
            if return_state:
                state = dataclases.replace(self, stack=stack)
            else:
                state = None
            if extra_outputs or state is not None:
                return ForwardResult(outputs, extra_outputs, state)
            else:
                return outputs

    def initial_state(self,
        batch_size: int,
        *args: Any,
        return_actions: bool=False,
        **kwargs: Any
    ) -> Unidirectional.State:
        return self.State(
            self,
            self.initial_stack(
                batch_size,
                *args,
                **kwargs
            ),
            return_actions
        )

    def initial_stack(self,
        batch_size: int,
        *args: Any,
        **kwargs: Any
    ) -> DifferentiableStack:
        raise NotImplementedError

    def next_stack(self,
        stack: DifferentiableStack,
        action_tensor: torch.Tensor,
        pushed_vector: torch.Tensor
    ) -> DifferentiableStack:
        raise NotImplementedError

    def transform_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions
