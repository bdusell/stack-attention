from collections.abc import Callable, Iterable
from typing import Any, Optional, Union

import torch

from .unidirectional import Unidirectional, ForwardResult

class SimpleUnidirectional(Unidirectional):
    r"""A sequential module that has no temporal recurrence, but applies some
    function to every timestep."""

    def forward_single(self,
        input_tensor: torch.Tensor,
        *args: Any,
        **kwargs: Any
    ) -> torch.Tensor:
        r"""Transform an input tensor for a single timestep.

        :param input_tensor: A tensor of size :math:`B \times \cdots`
            representing a tensor for a single timestep.
        :return: A tensor of size :math:`B \times cdots`.
        """
        raise NotImplementedError

    def forward_sequence(self,
        input_sequence: torch.Tensor,
        *args: Any,
        **kwargs: Any
    ) -> torch.Tensor:
        r"""Transform a sequence of tensors.

        :param input_sequence: A tensor of size :math:`B \times n \times \cdots`
            representing a sequence of tensors.
        :return: A tensor of size :math:`B \times n \cdots`.
        """
        raise NotImplementedError

    def initial_output(self,
        batch_size: int,
        *args: Any,
        **kwargs: Any
    ) -> torch.Tensor:
        r"""Get the output of the initial state. By default, this simply
        raises an error.

        :param batch_size: Batch size.
        :return: A tensor of size :math:`B \times \cdots`.
        """
        raise ValueError(
            'tried to get the output of the initial state of a '
            'SimpleUnidirectional, but the output is not defined'
        )

    def transform_args(self,
        args: list[Any],
        func: Callable[[torch.Tensor], torch.Tensor]
    ) -> list[Any]:
        return args

    def transform_kwargs(self,
        kwargs: dict[str, Any],
        func: Callable[[torch.Tensor], torch.Tensor]
    ) -> dict[str, Any]:
        return kwargs

    class State(Unidirectional.State):

        parent: 'SimpleUnidirectional'
        input_tensor: Optional[torch.Tensor]
        _batch_size: Optional[int]
        args: list[Any]
        kwargs: dict[str, Any]

        def __init__(self,
            parent: 'SimpleUnidirectional',
            input_tensor: Optional[torch.Tensor],
            batch_size: Optional[int],
            args: list[Any],
            kwargs: dict[str, Any]
        ):
            self.parent = parent
            self.input_tensor = input_tensor
            self._batch_size = batch_size
            self.args = args
            self.kwargs = kwargs

        def next(self, input_tensor: torch.Tensor) -> Unidirectional.State:
            return self.parent.State(self.parent, input_tensor, None, self.args, self.kwargs)

        def output(self) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
            if self._batch_size is not None:
                return self.parent.initial_output(self._batch_size, *self.args, **self.kwargs)
            else:
                return self.parent.forward_single(self.input_tensor, *self.args, **self.kwargs)

        def batch_size(self) -> int:
            if self._batch_size is not None:
                return self._batch_size
            else:
                return self.input_tensor.size(0)

        def transform_tensors(self,
            func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Unidirectional.State:
            if self.input_tensor is None:
                # TODO Simply returning self would not change the batch size.
                # It's possible to work around this by running func() on a
                # dummy tensor.
                raise ValueError(
                    'cannot call transform_tensors() on initial state of '
                    'SimpleUnidirectional'
                )
            else:
                return self.parent.State(
                    self.parent,
                    func(self.input_tensor),
                    None,
                    self.parent.transform_args(self.args, func),
                    self.parent.transform_kwargs(self.kwargs, func)
                )

        def fastforward(self, input_sequence: torch.Tensor) -> Unidirectional.State:
            if input_sequence.size(1) == 0:
                return self
            else:
                return self.next(input_sequence[:, -1])

        def outputs(self,
            input_sequence: torch.Tensor,
            include_first: bool
        ) -> Union[Iterable[torch.Tensor], Iterable[tuple[torch.Tensor, ...]]]:
            if include_first:
                first_output = self.output()
            output = self.parent.forward_sequence(input_sequence, *self.args, **self.kwargs)
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
                        input_sequence[:, -1],
                        self._batch_size,
                        self.args,
                        self.kwargs
                    )
                return ForwardResult(output, [], state)
            else:
                return output

    def initial_state(self,
        batch_size: int,
        *args: Any,
        **kwargs: Any
    ) -> Unidirectional.State:
        return self.State(self, None, batch_size, args, kwargs)

class SimpleLayerUnidirectional(SimpleUnidirectional):

    def __init__(self, func: Callable):
        super().__init__()
        self.func = func

    def forward_single(self,
        input_tensor: torch.Tensor,
        *args: Any,
        **kwargs: Any
    ) -> torch.Tensor:
        return self.func(input_tensor, *args, **kwargs)

    def forward_sequence(self,
        input_sequence: torch.Tensor,
        *args: Any,
        **kwargs: Any
    ) -> torch.Tensor:
        return self.func(input_sequence, *args, **kwargs)

class SimpleReshapingLayerUnidirectional(SimpleLayerUnidirectional):

    def forward_single(self,
        input_tensor: torch.Tensor,
        *args: Any,
        **kwargs: Any
    ) -> torch.Tensor:
        return self.func(input_tensor.unsqueeze(1), *args, **kwargs).squeeze(1)
