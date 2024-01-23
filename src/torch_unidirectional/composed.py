from collections.abc import Callable, Iterable, Mapping, Sequence
import dataclasses
import itertools
from typing import Any, Optional, Union

import torch

from .unidirectional import Unidirectional, ForwardResult
from .util import unwrap_output_tensor, ensure_is_forward_result

class ComposedUnidirectional(Unidirectional):
    """Stacks one undirectional model on another, so that the outputs of the
    first are fed as inputs to the second."""

    def __init__(self, first: Unidirectional, second: Unidirectional):
        super().__init__(first._tags | second._tags)
        self.first = first
        self.second = second

    def forward(self,
        input_sequence: torch.Tensor,
        *args: Any,
        initial_state: Optional[Unidirectional.State]=None,
        return_state: bool=False,
        include_first: bool=True,
        tag_kwargs=None,
        **kwargs: Any
    ) -> Union[torch.Tensor, ForwardResult]:
        if (args or kwargs) and 'main' not in self._tags:
            raise ValueError('this module does not accept extra args or kwargs')
        if initial_state is None and not return_state:
            first_args, first_kwargs = get_args(self.first, args, kwargs, tag_kwargs, include_first)
            second_args, second_kwargs = get_args(self.second, args, kwargs, tag_kwargs, include_first)
            first_result = ensure_is_forward_result(self.first(
                input_sequence,
                *first_args,
                return_state=False,
                **first_kwargs
            ))
            second_result = ensure_is_forward_result(self.second(
                first_result.output,
                *second_args,
                return_state=False,
                **second_kwargs
            ))
            return unwrap_output_tensor(ForwardResult(
                second_result.output,
                tuple(itertools.chain(first_result.extra_outputs, second_result.extra_outputs)),
                None
            ))
        else:
            return super().forward(
                input_sequence,
                *args,
                initial_state=initial_state,
                return_state=return_state,
                tag_kwargs=tag_kwargs,
                **kwargs
            )

    @dataclasses.dataclass
    class State(Unidirectional.State):

        first_state: Unidirectional.State
        second_state: Unidirectional.State

        def next(self, input_tensor: torch.Tensor) -> Unidirectional.State:
            new_first_state = self.first_state.next(input_tensor)
            first_output = new_first_state.output()
            if not isinstance(first_output, torch.Tensor):
                # TODO Handle extra outputs.
                raise TypeError
            new_second_state = self.second_state.next(first_output)
            return ComposedUnidirectional.State(new_first_state, new_second_state)

        def output(self) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
            return self.second_state.output()

        def detach(self) -> Unidirectional.State:
            return ComposedUnidirectional.State(
                self.first_state.detach(),
                self.second_state.detach()
            )

        def batch_size(self) -> int:
            return self.first_state.batch_size()

        def slice_batch(self, s: slice) -> Unidirectional.State:
            return ComposedUnidirectional.State(
                self.first_state.slice_batch(s),
                self.second_state.slice_batch(s)
            )

        def transform_tensors(self,
            func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Unidirectional.State:
            return ComposedUnidirectional.State(
                self.first_state.transform_tensors(func),
                self.second_state.transform_tensors(func)
            )

        def _get_first_outputs(self, input_sequence: torch.Tensor) -> ForwardResult:
            # TODO Handle extra outputs.
            return _ensure_outputs_are_tensor(self.first_state.outputs(
                input_sequence,
                include_first=False
            ))

        def fastforward(self, input_sequence: torch.Tensor) -> Unidirectional.State:
            return self.second_state.fastforward(self._get_first_outputs(input_sequence))

        def outputs(self,
            input_sequence: torch.Tensor,
            include_first: bool
        ) -> Union[Iterable[torch.Tensor], Iterable[tuple[torch.Tensor, ...]]]:
            return self.second_state.outputs(
                self._get_first_outputs(input_sequence),
                include_first=include_first
            )

        def forward(self,
            input_sequence: torch.Tensor,
            return_state: bool,
            include_first: bool
        ) -> Union[torch.Tensor, ForwardResult]:
            first_kwargs = dict(
                return_state=return_state,
                include_first=False
            )
            second_kwargs = dict(
                return_state=return_state,
                include_first=include_first
            )
            if return_state:
                first_result = self.first_state.forward(input_sequence, **first_kwargs)
                assert isinstance(first_result, ForwardResult)
                assert first_result.state is not None
                second_result = self.second_state.forward(first_result.output, **second_kwargs)
                assert isinstance(second_result, ForwardResult)
                assert second_result.state is not None
                second_result.extra_outputs = first_result.extra_outputs + second_result.extra_outputs
                second_result.state = ComposedUnidirectional.State(first_result.state, second_result.state)
                return second_result
            else:
                first_result = self.first_state.forward(input_sequence, **first_kwargs)
                if isinstance(first_result, ForwardResult):
                    first_output = first_result.output
                    first_extra_outputs = first_result.extra_outputs
                else:
                    first_output = first_result
                    first_extra_outputs = []
                second_result = self.second_state.forward(first_output, **second_kwargs)
                if isinstance(second_result, ForwardResult):
                    second_output = second_result.output
                    second_extra_outputs = second_result.extra_outputs
                else:
                    second_output = second_result
                    second_extra_outputs = []
                return unwrap_output_tensor(ForwardResult(
                    second_output,
                    first_extra_outputs + second_extra_outputs,
                    None
                ))

    def initial_state(self, batch_size, *args, tag_kwargs=None, **kwargs):
        if (args or kwargs) and 'main' not in self._tags:
            raise ValueError('this module does not accept extra args or kwargs')
        first_args, first_kwargs = get_args(self.first, args, kwargs, tag_kwargs, None)
        second_args, second_kwargs = get_args(self.second, args, kwargs, tag_kwargs, None)
        return self.State(
            self.first.initial_state(batch_size, *first_args, **first_kwargs),
            self.second.initial_state(batch_size, *second_args, **second_kwargs)
        )

def _ensure_outputs_are_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    else:
        return torch.stack(list(x), dim=1)

def get_args(module, args, kwargs, tag_kwargs, include_first):
    result_args = []
    result_kwargs = dict(include_first=False) if include_first is not None else {}
    if 'main' in module._tags:
        result_args.extend(args)
        if include_first is not None:
            result_kwargs['include_first'] = include_first
        result_kwargs.update(kwargs)
    if tag_kwargs:
        if isinstance(module, ComposedUnidirectional):
            result_kwargs['tag_kwargs'] = tag_kwargs
        else:
            # TODO The ordering might not be deterministic.
            for tag in module._tags:
                if tag in tag_kwargs:
                    result_kwargs.update(tag_kwargs[tag])
    return result_args, result_kwargs
