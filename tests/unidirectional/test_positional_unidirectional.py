import torch

from torch_unidirectional import PositionalUnidirectional

class MyPositional(PositionalUnidirectional):

    def forward_from_position(self, input_sequence, position):
        indexes = torch.arange(
            position,
            position + input_sequence.size(1),
            device=input_sequence.device
        )
        return input_sequence + indexes[None]

    def forward_at_position(self, input_tensor, position):
        return input_tensor + position

def test_forward_matches_iterative():
    model = MyPositional()
    batch_size = 5
    sequence_length = 13
    generator = torch.manual_seed(123)
    input_sequence = torch.rand((batch_size, sequence_length), generator=generator)
    forward_output = model(input_sequence, include_first=False)
    assert forward_output.size() == (batch_size, sequence_length)
    state = model.initial_state(batch_size)
    for i in range(sequence_length):
        input_tensor = input_sequence[:, i]
        state = state.next(input_tensor)
        output = state.output()
        assert output.size() == (batch_size,)
        torch.testing.assert_close(output, forward_output[:, i])

def test_fastforward_and_include_first():
    model = MyPositional()
    batch_size = 5
    sequence_length = 13
    generator = torch.manual_seed(123)
    input_sequence = torch.rand((batch_size, sequence_length), generator=generator)
    forward_output = model(input_sequence, include_first=False)
    assert forward_output.size() == (batch_size, sequence_length)
    state = model.initial_state(batch_size)
    pos = 5
    state = state.fastforward(input_sequence[:, :pos])
    output = state.output()
    assert output.size() == (batch_size,)
    torch.testing.assert_close(output, forward_output[:, pos-1])
    include_first_output = state.forward(
        input_sequence[:, pos:],
        return_state=False,
        include_first=True
    )
    assert include_first_output.size() == (batch_size, sequence_length - pos + 1)
    torch.testing.assert_close(include_first_output, forward_output[:, pos-1:])
