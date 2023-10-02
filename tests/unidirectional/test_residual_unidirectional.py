import torch

from torch_unidirectional import ResidualUnidirectional, PositionalUnidirectional

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

    def initial_state(self, *args, alpha, beta, **kwargs):
        assert alpha == 123
        assert beta == 'moo'
        return super().initial_state(*args, **kwargs)

def test_forward_matches_iterative():
    batch_size = 5
    sequence_length = 13
    input_size = 7
    generator = torch.manual_seed(123)
    alpha = 123
    beta = 'moo'
    wrapped_model = AdditivePositional()
    model = ResidualUnidirectional(wrapped_model)
    input_sequence = torch.rand((batch_size, sequence_length, input_size), generator=generator)
    expected_forward_output = (
        input_sequence +
        wrapped_model(input_sequence, include_first=False, alpha=alpha, beta=beta)
    )
    assert expected_forward_output.size() == (batch_size, sequence_length, input_size)
    forward_output = model(input_sequence, include_first=False, alpha=alpha, beta=beta)
    assert forward_output.size() == (batch_size, sequence_length, input_size)
    torch.testing.assert_close(forward_output, expected_forward_output)
    state = model.initial_state(batch_size, alpha=alpha, beta=beta)
    for i in range(sequence_length):
        input_tensor = input_sequence[:, i]
        state = state.next(input_tensor)
        output = state.output()
        assert output.size() == (batch_size, input_size)
        torch.testing.assert_close(output, forward_output[:, i])
