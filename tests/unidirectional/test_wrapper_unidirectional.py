import torch

from torch_unidirectional import (
    SimpleWrapperUnidirectional,
    PositionalUnidirectional,
    SimpleLayerUnidirectional
)

class AdditivePositional(PositionalUnidirectional):

    def forward_from_position(self, input_sequence, position):
        indexes = torch.arange(
            position,
            position + input_sequence.size(1),
            device=input_sequence.device
        )
        return input_sequence + indexes[None, :, None]

    def forward_at_position(self, input_tensor, position):
        return input_tensor + position

    def initial_state(self, *args, **kwargs):
        # TODO add custom kwarg
        return super().initial_state(*args, **kwargs)

def test_forward_matches_iterative():
    batch_size = 5
    sequence_length = 13
    input_size = 7
    generator = torch.manual_seed(123)
    wrapped_model = AdditivePositional()
    def func(x, f):
        return x + f(2 * x)
    model = SimpleWrapperUnidirectional(wrapped_model, func)
    for p in model.parameters():
        p.data.uniform_(-1.0, 1.0, generator=generator)
    input_sequence = torch.rand((batch_size, sequence_length, input_size), generator=generator)
    expected_forward_output = (
        input_sequence +
        wrapped_model(2 * input_sequence, include_first=False)
    )
    assert expected_forward_output.size() == (batch_size, sequence_length, input_size)
    forward_output = model(input_sequence, include_first=False)
    assert forward_output.size() == (batch_size, sequence_length, input_size)
    torch.testing.assert_close(forward_output, expected_forward_output)
    state = model.initial_state(batch_size)
    for i in range(sequence_length):
        input_tensor = input_sequence[:, i]
        state = state.next(input_tensor)
        output = state.output()
        assert output.size() == (batch_size, input_size)
        torch.testing.assert_close(output, forward_output[:, i])
