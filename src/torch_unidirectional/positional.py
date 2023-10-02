from collections.abc import Callable, Iterable
from typing import Any, Optional, Union

import torch

from .unidirectional import Unidirectional, ForwardResult

class PositionalUnidirectional(Unidirectional):

    def forward_from_position(self,
        input_sequence: torch.Tensor,
        position: int
    ) -> torch.Tensor:
        r"""Compute the outputs for a sequence of inputs, starting at a certain
        position.

        :param input_sequence: A tensor of size :math:`B \times n \times \cdots`
            representing a sequence of input tensors.
        :param position: An index indicating the timestep corresponding to the
            first input of ``input_sequence``. The first timestep has index 0.
        :return: A tensor of size :math:`B \times n' \times \cdots`
            representing a sequence of output tensors.
        """
        raise NotImplementedError

    def forward_at_position(self,
        input_tensor: torch.Tensor,
        position: int
    ) -> torch.Tensor:
        r"""Compute the output for a single input at a certain position.

        :param input_tensor: A tensor of size :math:`B \times cdots`
            representing an input tensor for a single timestep.
        :param position: An index indicating the current timestep. The first
            timestep has index 0.
        :return: A tensor of size :math:`B \times \cdots` representing the
            output tensor corresponding to the input tensor.
        """
        raise NotImplementedError

    class State(Unidirectional.State):

        parent: 'PositionalUnidirectional'
        position: int
        input_tensor: Optional[torch.Tensor]

        def __init__(self,
            parent: 'PositionalUnidirectional',
            position: int,
            input_tensor: Optional[torch.Tensor]
        ):
            super().__init__()
            self.parent = parent
            self.position = position
            self._input_tensor = input_tensor

        def next(self, input_tensor: torch.Tensor) -> Unidirectional.State:
            return self.parent.State(
                self.parent,
                self.position + 1,
                input_tensor
            )

        def output(self) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
            if self._input_tensor is None:
                raise ValueError(
                    'initial state of PositionalUnidirectional does not have '
                    'an output'
                )
            return self.parent.forward_at_position(self._input_tensor, self.position - 1)

        def batch_size(self) -> int:
            if self._input_tensor is None:
                raise ValueError(
                    'initial state of PositionalUnidirectional does not have '
                    'a batch size'
                )
            return self._input_tensor.size(0)

        def transform_tensors(self,
            func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Unidirectional.State:
            if self._input_tensor is None:
                return self
            else:
                return self.parent.State(
                    self.parent,
                    self.position,
                    func(self._input_tensor)
                )

        def fastforward(self, input_sequence: torch.Tensor) -> Unidirectional.State:
            length = input_sequence.size(1)
            if length == 0:
                return self
            else:
                return self.parent.State(
                    self.parent,
                    self.position + length,
                    input_sequence[:, -1]
                )

        def outputs(self,
            input_sequence: torch.Tensor,
            include_first: bool
        ) -> Union[Iterable[torch.Tensor], Iterable[tuple[torch.Tensor, ...]]]:
            if include_first:
                # NOTE Another way to include the first output would be to
                # decrement the position by 1.
                first_output = self.output()
            output = self.parent.forward_from_position(input_sequence, self.position)
            if include_first:
                output = torch.concat([first_output[:, None], output], dim=1)
            return output

        def forward(self,
            input_sequence: torch.Tensor,
            return_state: bool,
            include_first: bool
        ) -> Union[torch.Tensor, ForwardResult]:
            output = self.outputs(input_sequence, include_first)
            if return_state:
                if input_sequence.size(1) == 0:
                    state = self
                else:
                    state = self.parent.State(
                        self.parent,
                        self.position + input_sequence.size(1),
                        input_sequence[:, -1]
                    )
                return ForwardResult(output, [], state)
            else:
                return output

    def initial_state(self,
        batch_size: int,
        *args: Any,
        **kwargs: Any
    ) -> Unidirectional.State:
        return self.State(self, 0, None)
