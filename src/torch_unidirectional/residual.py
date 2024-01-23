from collections.abc import Callable, Iterable
import dataclasses
from typing import Any, Optional, Union

import torch

from .unidirectional import Unidirectional, ForwardResult
from .util import unwrap_output_tensor, ensure_is_forward_result

class ResidualUnidirectional(Unidirectional):

    def __init__(self, module: Unidirectional):
        super().__init__()
        self.wrapped_module = module

    def forward(self,
        input_sequence: torch.Tensor,
        initial_state: Optional[Unidirectional.State]=None,
        return_state: bool=False,
        include_first: bool=True,
        **kwargs: Any
    ) -> Union[torch.Tensor, ForwardResult]:
        if initial_state is None and not return_state and not include_first:
            wrapped_result = ensure_is_forward_result(self.wrapped_module(
                input_sequence,
                initial_state=None,
                return_state=return_state,
                include_first=include_first,
                **kwargs
            ))
            return unwrap_output_tensor(ForwardResult(
                input_sequence + wrapped_result.output,
                wrapped_result.extra_outputs,
                None
            ))
        else:
            return super().forward(
                input_sequence,
                initial_state=initial_state,
                return_state=return_state,
                include_first=include_first,
                **kwargs
            )

    @dataclasses.dataclass
    class State(Unidirectional.State):

        input_tensor: Optional[torch.Tensor]
        wrapped_state: Unidirectional.State

        def next(self, input_tensor: torch.Tensor) -> Unidirectional.State:
            return ResidualUnidirectional.State(
                input_tensor,
                self.wrapped_state.next(input_tensor)
            )

        def output(self) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
            # TODO Handle multiple outputs
            return self._get_input_tensor() + self.wrapped_state.output()

        def _get_input_tensor(self) -> torch.Tensor:
            if self.input_tensor is None:
                raise ValueError(
                    'the initial state of a ResidualUnidirectional has no '
                    'input, so it has no output'
                )
            return self.input_tensor

        def batch_size(self) -> int:
            return self.wrapped_state.batch_size()

        def transform_tensors(self,
            func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Unidirectional.State:
            return ResidualUnidirectional.State(
                func(self.input_tensor) if self.input_tensor is not None else None,
                self.wrapped_state.transform_tensors(func)
            )

        def fastforward(self, input_sequence: torch.Tensor) -> Unidirectional.State:
            if input_sequence.size(1) == 0:
                return self
            else:
                return ResidualUnidirectional.State(
                    input_sequence[:, -1],
                    self.wrapped_state.fastforward(input_sequence)
                )

        def states(self,
            input_sequence: torch.Tensor,
            include_first: bool
        ) -> Iterable[Unidirectional.State]:
            # TODO Implement this efficiently.
            raise NotImplementedError

        def outputs(self,
            input_sequence: torch.Tensor,
            include_first: bool
        ) -> Union[Iterable[torch.Tensor], Iterable[tuple[torch.Tensor, ...]]]:
            if include_first:
                first_input = self._get_input_tensor()
            output = self.wrapped_state.outputs(input_sequence, include_first)
            if include_first:
                input_sequence = torch.concat([first_input[:, None], input_sequence], dim=1)
            return input_sequence + output

        def forward(self,
            input_sequence: torch.Tensor,
            return_state: bool,
            include_first: bool
        ) -> Union[torch.Tensor, ForwardResult]:
            if return_state:
                # TODO
                raise NotImplementedError
            else:
                return self.outputs(input_sequence, include_first)

    def initial_state(self,
        batch_size: int,
        *args: Any,
        **kwargs: Any
    ) -> Unidirectional.State:
        return self.State(
            None,
            self.wrapped_module.initial_state(batch_size, *args, **kwargs)
        )
