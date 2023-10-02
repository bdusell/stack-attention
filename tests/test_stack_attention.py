import torch

from torch_unidirectional import SimpleLayerUnidirectional
from stack_attention.unidirectional_encoder_layer import (
    get_unidirectional_encoder_layer_with_custom_attention
)
from stack_attention.superposition import SuperpositionStackAttention
from stack_attention.nondeterministic import NondeterministicStackAttention

class CustomModule(torch.nn.Module):

    def forward(self, input_tensor, alpha, beta):
        assert alpha == 123
        assert beta == 'asdf'
        return input_tensor

def test_custom_encoder_layer():
    d_model = 7
    batch_size = 5
    sequence_length = 13
    attention_func = SimpleLayerUnidirectional(CustomModule())
    alpha = 123
    beta = 'asdf'
    generator = torch.manual_seed(123)
    input_sequence = torch.rand((batch_size, sequence_length, d_model), generator=generator)
    model = get_unidirectional_encoder_layer_with_custom_attention(
        attention_func,
        d_model=d_model,
        feedforward_size=17,
        dropout=None,
        tag='custom_attention'
    )
    for p in model.parameters():
        p.data.uniform_(generator=generator)
    forward_output = model(input_sequence, include_first=False, tag_kwargs=dict(
        custom_attention=dict(
            alpha=alpha,
            beta=beta
        )
    ))
    assert forward_output.size() == (batch_size, sequence_length, d_model)
    state = model.initial_state(batch_size, tag_kwargs=dict(
        custom_attention=dict(
            alpha=alpha,
            beta=beta
        )
    ))
    for i in range(sequence_length):
        input_tensor = input_sequence[:, i]
        state = state.next(input_tensor)
        output = state.output()
        assert output.size() == (batch_size, d_model)
        torch.testing.assert_close(output, forward_output[:, i])

def test_superposition_stack_attention():
    d_model = 7
    batch_size = 5
    sequence_length = 13
    stack_embedding_size = d_model
    generator = torch.manual_seed(123)
    input_sequence = torch.rand((batch_size, sequence_length, d_model), generator=generator)
    model = SuperpositionStackAttention(d_model, stack_embedding_size)
    for p in model.parameters():
        p.data.uniform_(generator=generator)
    forward_output = model(input_sequence, include_first=False)
    assert forward_output.size() == (batch_size, sequence_length, d_model)
    state = model.initial_state(batch_size)
    for i in range(sequence_length):
        input_tensor = input_sequence[:, i]
        state = state.next(input_tensor)
        output = state.output()
        assert output.size() == (batch_size, d_model)
        torch.testing.assert_close(output, forward_output[:, i])

def test_nondeterministic_stack_attention():
    d_model = 7
    batch_size = 5
    sequence_length = 13
    stack_embedding_size = d_model
    generator = torch.manual_seed(123)
    input_sequence = torch.rand((batch_size, sequence_length, d_model), generator=generator)
    model = NondeterministicStackAttention(
        d_model=d_model,
        num_states=3,
        stack_alphabet_size=4,
        stack_embedding_size=stack_embedding_size
    )
    for p in model.parameters():
        p.data.uniform_(generator=generator)
    forward_output = model(input_sequence, include_first=False, block_size=7)
    assert forward_output.size() == (batch_size, sequence_length, d_model)
    state = model.initial_state(batch_size, sequence_length=sequence_length, block_size=13)
    for i in range(sequence_length):
        input_tensor = input_sequence[:, i]
        state = state.next(input_tensor)
        output = state.output()
        assert output.size() == (batch_size, d_model)
        torch.testing.assert_close(output, forward_output[:, i], atol=1e-5, rtol=1e-5)
