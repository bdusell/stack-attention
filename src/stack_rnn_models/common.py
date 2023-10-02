from collections.abc import Callable
from typing import Optional

import torch

from torch_unidirectional import Unidirectional
from torch_extras.layer import FeedForward

def cached_tensor(method):
    name = method.__name__
    def wrapped_method(self):
        result = self._cached_tensors.get(name)
        if result is None:
            result = self._cached_tensors[name] = method(self)
        return result
    return wrapped_method

class StackRNNBase(Unidirectional):

    def __init__(self,
        input_size: int,
        stack_reading_size: int,
        controller: Callable[[int], torch.nn.Module],
        controller_output_size: int,
        reading_layer_sizes: Optional[list[Optional[int]]]=None,
        include_reading_in_output: bool=False
    ):
        """
        :param stack_reading_size: The size of the reading vector returned
            from the stack module.
        :param controller: A constructor function that takes an input size and
            returns a Unidirectional implementing the controller.
        :param reading_layer_sizes: An optional list specifying the sizes of
            hidden layers to apply between the stack reading and the
            controller. A size of ``None`` indicates that the
            ``stack_reading_size`` should be used.
        :param include_reading_in_output: Whether to include the stack reading
            (after applying any layers) in the output along with the output of
            the controller (concatenated into one vector).
        """
        super().__init__()
        self._stack_reading_size = stack_reading_size
        if reading_layer_sizes:
            reading_layer_sizes = [x if x is not None else stack_reading_size for x in reading_layer_sizes]
            self.reading_layer = FeedForward(
                input_size=stack_reading_size,
                layer_sizes=reading_layer_sizes,
                activation=torch.nn.Tanh()
            )
            reading_layer_output_size = self.reading_layer.output_size()
        else:
            self.reading_layer = torch.nn.Identity()
            reading_layer_output_size = stack_reading_size
        self.include_reading_in_output = include_reading_in_output
        self.controller = controller(input_size + reading_layer_output_size)
        self._output_size = controller_output_size
        if self.include_reading_in_output:
            self._output_size += reading_layer_output_size

    def output_size(self) -> int:
        return self._output_size

    def forward(self, input_sequence, *args, **kwargs):
        # Automatically add the length of the input as an extra argument.
        sequence_length = input_sequence.size(1)
        return super().forward(
            input_sequence,
            *args,
            sequence_length=sequence_length,
            **kwargs)

    class State(Unidirectional.State):

        def __init__(self, rnn, num_inputs_read, sequence_length, hidden_state,
                previous_stack, return_actions, return_readings, stack_args,
                stack_kwargs):
            super().__init__()
            self.rnn = rnn
            self.num_inputs_read = num_inputs_read
            self.sequence_length = sequence_length
            self.hidden_state = hidden_state
            self.previous_stack = previous_stack
            self.return_actions = return_actions
            self.return_readings = return_readings
            self.stack_args = stack_args
            self.stack_kwargs = stack_kwargs
            self._cached_tensors = {}

        def next(self, input_tensor):
            stack = self.get_stack()
            reading_layer_output = self.get_reading_layer_output()
            controller_input = torch.cat((input_tensor, reading_layer_output), dim=1)
            next_hidden_state = self.hidden_state.next(controller_input)
            return self.rnn.State(
                rnn=self.rnn,
                num_inputs_read=self.num_inputs_read + 1,
                sequence_length=self.sequence_length,
                hidden_state=next_hidden_state,
                previous_stack=stack,
                return_actions=self.return_actions,
                return_readings=self.return_readings,
                stack_args=None,
                stack_kwargs=None
            )

        def output(self):
            output = self.get_output()
            extras = []
            if self.return_actions:
                if self.num_inputs_read < self.sequence_length:
                    actions = self.get_actions()
                else:
                    actions = None
                extras.append(actions)
            if self.return_readings:
                if self.num_inputs_read < self.sequence_length:
                    reading = self.get_reading()
                else:
                    reading = None
                extras.append(reading)
            if extras:
                return (output, *extras)
            else:
                return output

        def get_actions(self):
            stack, actions = self.get_stack_and_actions()
            return actions

        def get_stack(self):
            stack, actions = self.get_stack_and_actions()
            return stack

        @cached_tensor
        def get_stack_and_actions(self):
            if self.previous_stack is None:
                stack = self.rnn.initial_stack(
                    self.hidden_state.batch_size(),
                    self.rnn._stack_reading_size,
                    self.sequence_length,
                    *self.stack_args,
                    **self.stack_kwargs
                )
                actions = None
            else:
                stack, actions = self.compute_stack(
                    self.get_hidden_state_output(),
                    self.previous_stack
                )
                # The previous stack is no longer needed now.
                self.previous_stack = None
            return stack, actions

        @cached_tensor
        def get_reading(self):
            return self.get_stack().reading()

        @cached_tensor
        def get_reading_layer_output(self):
            return self.rnn.reading_layer(self.get_reading())

        @cached_tensor
        def get_hidden_state_output(self):
            return self.hidden_state.output()

        @cached_tensor
        def get_output(self):
            if self.rnn.include_reading_in_output:
                return torch.concat([
                    self.get_hidden_state_output(),
                    self.get_reading_layer_output()
                ], dim=1)
            else:
                return self.get_hidden_state_output()

        def detach(self):
            return self.rnn.State(
                rnn=self.rnn,
                num_inputs_read=self.num_inputs_read,
                sequence_length=self.sequence_length,
                hidden_state=self.hidden_state.detach(),
                previous_stack=self.previous_stack.detach() if self.previous_stack is not None else None,
                return_actions=self.return_actions,
                return_readings=self.return_readings,
                stack_args=self.stack_args,
                stack_kwargs=self.stack_kwargs
            )

        def batch_size(self):
            return self.hidden_state.batch_size()

        def slice_batch(self, s):
            return self.rnn.State(
                rnn=self.rnn,
                num_inputs_read=self.num_inputs_read,
                sequence_length=self.sequence_length,
                hidden_state=self.hidden_state.slice_batch(s),
                previous_stack=self.previous_stack.slice_batch(s) if self.previous_stack is not None else None,
                return_actions=self.return_actions,
                return_readings=self.return_readings,
                stack_args=self.stack_args,
                stack_kwargs=self.stack_kwargs
            )

        def compute_stack(self, hidden_state, stack):
            raise NotImplementedError

    def initial_state(self,
            batch_size: int,
            *args,
            sequence_length: Optional[int]=None,
            return_actions: bool=False,
            return_readings: bool=False,
            first_layer: Optional[torch.Tensor]=None,
            **kwargs):
        """Get the initial state of the stack RNN.

        :param sequence_length: Used to determine when the last timestep is
            reached, which may avoid some unnecessary computation. The actions
            and reading will not be returned for the last timestep if they are
            not needed.
        :param return_actions: If true, then the output at each timestep will
            also include the stack actions that were emitted at that timestep.
            Note that the stack actions for timestep 0 are always ``None``.
        :param return_readings: If true, then the output at each timestep will
            also include the stack reading that was emitted just before the
            current timestep.
        :param first_layer: Will be passed to the controller.
        :param args: Will be passed to :py:meth:`initial_stack`.
        :param kwargs: Will be passed to :py:meth:`initial_stack`.
        """
        return self.State(
            rnn=self,
            num_inputs_read=0,
            sequence_length=sequence_length,
            hidden_state=self.controller.initial_state(
                batch_size,
                first_layer=first_layer
            ),
            # There is no "previous stack" for the initial hidden state, so
            # set it to None. It will call initial_stack() to supply the stack
            # for the next timestep.
            previous_stack=None,
            return_actions=return_actions,
            return_readings=return_readings,
            stack_args=args,
            stack_kwargs=kwargs
        )

    def initial_stack(self, batch_size, reading_size, sequence_length, *args, **kwargs):
        raise NotImplementedError
