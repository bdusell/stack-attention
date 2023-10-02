from collections.abc import Callable, Iterable
import dataclasses
from typing import Any, Optional, Union

import more_itertools
import torch

@dataclasses.dataclass
class ForwardResult:
    r"""The output of a call to :py:meth:`Unidirectional.forward` or
    :py:meth:`Unidirectional.State.forward`."""

    output: torch.Tensor
    r"""The main output tensor of the module."""
    extra_outputs: list[list[Any]]
    r"""A list of extra outputs returned alongside the main output."""
    state: 'Optional[Unidirectional.State]'
    r"""An optional state representing the updated state of the module after
    reading the inputs."""

class Unidirectional(torch.nn.Module):
    """An API for unidirectional sequential neural networks (including RNNs
    and transformer decoders).

    Let :math:`B` be batch size, and :math:`n` be the length of the input
    sequence.
    """

    def __init__(self, tags=None):
        super().__init__()
        self._tags = tags if tags is not None else set()

    def forward(self,
        input_sequence: torch.Tensor,
        *args: Any,
        initial_state: Optional['Unidirectional.State']=None,
        return_state: bool=False,
        include_first: bool=True,
        **kwargs: Any
    ) -> Union[torch.Tensor, ForwardResult]:
        r"""Run this module on an entire sequence of inputs all at once.

        This can often be done more efficiently than processing each input one
        by one.

        :param input_sequence: A :math:`B \times n \times \cdots` tensor
            representing a sequence of :math:`n` input tensors.
        :param initial_state: An optional initial state to use instead of the
            default initial state created by :py:meth:`initial_state`.
        :param return_state: Whether to return the last :py:class:`State` of
            the module as an additional output. This state can be used to
            initialize a subsequent run.
        :param include_first: Whether to prepend an extra tensor to the
            beginning of the output corresponding to a prediction for the
            first element in the input. If ``include_first`` is true, then the
            length of the output tensor will be :math:`n + 1`. Otherwise, it
            will be :math:`n`.
        :param args: Extra arguments passed to :py:meth:`initial_state`.
        :param kwargs: Extra arguments passed to :py:meth:`initial_state`.
        :return: A :py:class:`~torch.Tensor` or a :py:class:`ForwardResult` that
            contains the output tensor. The output tensor will be of size
            :math:`B \times n+1 \times \cdots` if ``include_first`` is true and
            :math:`B \times n \times \cdots` otherwise. If
            :py:meth:`Unidirectional.State.output` returns extra outputs at
            each timestep, then they will be aggregated over all timesteps and
            returned as :py:class:`list`\ s in :py:attr:`ForwardResult.extra_outputs`.
            If ``return_state`` is true, then the final :py:class:`State` will
            be returned in :py:attr:`ForwardResult.state`. If there are no extra
            outputs and there is no state to return, just the output tensor is
            returned.
        """
        # input_sequence: B x n x ...
        if initial_state is not None:
            if not isinstance(initial_state, self.State):
                raise TypeError(f'initial_state must be of type {self.State.__name__}')
            state = initial_state
        else:
            batch_size = input_sequence.size(0)
            state = self.initial_state(batch_size, *args, **kwargs)
        # return : B x n x ...
        return state.forward(
            input_sequence,
            return_state=return_state,
            include_first=include_first
        )

    def __or__(self, other: 'Unidirectional') -> 'Unidirectional':
        r"""The ``|`` operator is overridden to compose two Unidirectionals."""
        from .composed import ComposedUnidirectional
        return ComposedUnidirectional(self, other)

    class State:
        """Represents the hidden state of the module after processing a certain
        number of inputs."""

        def next(self, input_tensor: torch.Tensor) -> 'Unidirectional.State':
            r"""Feed an input to this hidden state and produce the next hidden
            state.

            :param input_tensor: A tensor of size :math:`B \times \cdots`,
                representing an input for a single timestep.
            """
            raise NotImplementedError

        def output(self) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
            r"""Get the output associated with this state.

            For example, this can be the hidden state vector itself, or the
            hidden state passed through an affine transformation.

            The return value is either a tensor or a tuple whose first element
            is a tensor. The other elements of the tuple can be used to return
            extra outputs.

            :return: A :math:`B \times \cdots` tensor, or a tuple whose first
                element is a tensor. The other elements of the tuple can
                contain extra outputs. If there are any extra outputs, then
                the output of :py:meth:`forward` and
                :py:meth:`Unidirectional.forward` will contain the same
                number of extra outputs, where each extra output is a
                :py:class:`list` containing all the outputs across all
                timesteps.
            """
            raise NotImplementedError

        def detach(self) -> 'Unidirectional.State':
            """Return a copy of this state with all tensors detached."""
            return self.transform_tensors(lambda x: x.detach())

        def batch_size(self) -> int:
            """Get the batch size of the tensors in this state."""
            raise NotImplementedError

        def slice_batch(self, s: slice) -> 'Unidirectional.State':
            """Return a copy of this state with only certain batch elements
            included, determined by the slice ``s``.
            
            :param s: The slice object used to determine which batch elements
                to keep.
            """
            return self.transform_tensors(lambda x: x[s, ...])

        def transform_tensors(self,
            func: Callable[[torch.Tensor], torch.Tensor]
        ) -> 'Unidirectional.State':
            """Return a copy of this state with all tensors passed through a
            function.

            :param func: A function that will be applied to all tensors in this
                state.
            """
            raise NotImplementedError

        def fastforward(self, input_sequence: torch.Tensor) -> 'Unidirectional.State':
            r"""Feed a sequence of inputs to this state and return the
            resulting state.

            :param input_sequence: A :math:`B \times n \times \cdots` tensor,
                representing :math:`n` input tensors.
            :return: Updated state after reading ``input_sequence``.
            """
            state = self
            for input_tensor in input_sequence.transpose(0, 1):
                state = state.next(input_tensor)
            return state

        def states(self,
            input_sequence: torch.Tensor,
            include_first: bool
        ) -> Iterable['Unidirectional.State']:
            r"""Feed a sequence of inputs to this state and generate all the
            states produced after each input.

            :param input_sequence: A :math:`B \times n \times \cdots` tensor,
                representing :math:`n` input tensors.
            :param include_first: Whether to include ``self`` as the first
                state in the returned sequence of states.
            :return: Sequence of states produced by reading ``input_sequence``.
            """
            state = self
            if include_first:
                yield state
            for input_tensor in input_sequence.transpose(0, 1):
                state = state.next(input_tensor)
                yield state

        def outputs(self,
            input_sequence: torch.Tensor,
            include_first: bool
        ) -> Union[Iterable[torch.Tensor], Iterable[tuple[torch.Tensor, ...]]]:
            r"""Like :py:meth:`states`, but return the states' outputs.

            :param input_sequence: A :math:`B \times n \times \cdots` tensor,
                representing :math:`n` input tensors.
            :param include_first: Whether to include the output of ``self`` as
                the first output.
            """
            for state in self.states(input_sequence, include_first):
                yield state.output()

        def forward(self,
            input_sequence: torch.Tensor,
            return_state: bool,
            include_first: bool
        ) -> Union[torch.Tensor, ForwardResult]:
            r"""Like :py:meth:`Unidirectional.forward`, but start with this
            state as the initial state.

            This can often be done more efficiently than using :py:meth:`next`
            iteratively.

            :param input_sequence: A :math:`B \times n \times \cdots` tensor,
                representing :math:`n` input tensors.
            :param return_state: Whether to return the last :py:class:`State`
                of the module.
            :param include_first: Whether to prepend an extra tensor to the
                beginning of the output corresponding to a prediction for the
                first element in the input.
            :return: See :py:meth:`Unidirectional.forward`.
            """
            if return_state:
                outputs = []
                for state in self.states(input_sequence, include_first):
                    outputs.append(state.output())
                result = _stack_outputs(outputs)
                result.state = state
                return result
            else:
                return _unwrap_output_tensor(_stack_outputs(self.outputs(input_sequence, include_first)))

    def initial_state(self,
        batch_size: int,
        *args: Any,
        **kwargs: Any
    ) -> 'Unidirectional.State':
        r"""Get the initial state of the RNN.

        :param batch_size: Batch size.
        :param args: Extra arguments passed from :py:meth:`forward`.
        :param kwargs: Extra arguments passed from :py:meth:`forward`.
        :return: A state.
        """
        raise NotImplementedError

    def tag(self, tag):
        self._tags.add(tag)
        return self

    def main(self):
        return self.tag('main')

def _stack_outputs(
    outputs: Iterable[Union[torch.Tensor, tuple[torch.Tensor, ...]]]
) -> ForwardResult:
    it = iter(outputs)
    first = next(it)
    if isinstance(first, tuple):
        output, *extra = first
        output_list = [output]
        extra_lists = [[e] for e in extra]
        for output_t in it:
            if not isinstance(output_t, tuple):
                raise TypeError
            output, *extra = output_t
            if not isinstance(output, torch.Tensor):
                raise TypeError
            output_list.append(output)
            for extra_list, extra_item in more_itertools.zip_equal(extra_lists, extra):
                extra_list.append(extra_item)
    elif isinstance(first, torch.Tensor):
        output_list = [first]
        for output_t in it:
            if not isinstance(output_t, torch.Tensor):
                raise TypeError
            output_list.append(output_t)
        extra_lists = []
    else:
        raise TypeError
    output_tensor = torch.stack(output_list, dim=1)
    return ForwardResult(output_tensor, extra_lists, None)

def _unwrap_output_tensor(result: ForwardResult) -> Union[torch.Tensor, ForwardResult]:
    if result.extra_outputs or result.state is not None:
        return result
    else:
        return result.output
