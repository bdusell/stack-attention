from typing import Any, Optional, Union

import torch
from torch_semiring_einsum import AutomaticBlockSize

from torch_unidirectional import ForwardResult
from lib.semiring import log
from stack_rnn_models.stack import DifferentiableStack
from stack_rnn_models.vector_nondeterministic_stack import get_vector_nondeterministic_stack
from stack_rnn_models.nondeterministic_stack import logits_to_actions
from .stack_attention import StackAttention

class NondeterministicStackAttention(StackAttention):

    def __init__(self,
        d_model: int,
        num_states: int,
        stack_alphabet_size: int,
        stack_embedding_size: int
    ):
        Q = num_states
        S = stack_alphabet_size
        super().__init__(
            d_model,
            num_actions=Q*S*Q*(S+S+1),
            pushed_vector_size=stack_embedding_size,
            stack_reading_size=Q*S*stack_embedding_size
        )
        self.num_states = num_states
        self.stack_alphabet_size = stack_alphabet_size
        self.stack_embedding_size = stack_embedding_size
        self.bottom_vector_logits = torch.nn.Parameter(torch.zeros(stack_embedding_size))

    def forward(self,
        input_sequence: torch.Tensor,
        *args,
        **kwargs
    ) -> Union[torch.Tensor, ForwardResult]:
        return super().forward(
            input_sequence,
            sequence_length=input_sequence.size(1),
            *args,
            **kwargs
        )

    def initial_stack(self,
        batch_size: int,
        *args: Any,
        sequence_length: Optional[int]=None,
        block_size: Union[int, AutomaticBlockSize],
        **kwargs: Any
    ) -> DifferentiableStack:
        tensor = next(self.parameters())
        return get_vector_nondeterministic_stack(
            batch_size=batch_size,
            num_states=self.num_states,
            stack_alphabet_size=self.stack_alphabet_size,
            stack_embedding_size=self.stack_embedding_size,
            sequence_length=sequence_length,
            bottom_vector=torch.nn.functional.logsigmoid(self.bottom_vector_logits),
            block_size=block_size,
            dtype=tensor.dtype,
            device=tensor.device,
            semiring=log
        )

    def next_stack(self,
        stack: DifferentiableStack,
        action_tensor: torch.Tensor,
        pushed_vector: torch.Tensor
    ) -> DifferentiableStack:
        push, repl, pop = logits_to_actions(
            action_tensor,
            num_states=self.num_states,
            stack_alphabet_size=self.stack_alphabet_size,
            normalize=False
        )
        stack.update(push, repl, pop, pushed_vector)
        return stack
