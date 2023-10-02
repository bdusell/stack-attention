import torch

from torch_unidirectional import DropoutUnidirectional

def test_forward_and_iterative():
    batch_size = 5
    sequence_length = 13
    input_size = 7
    generator = torch.manual_seed(123)
    model = DropoutUnidirectional(0.2)
    input_sequence = torch.rand((batch_size, sequence_length, input_size), generator=generator)
    forward_output = model(input_sequence, include_first=False)
    assert forward_output.size() == (batch_size, sequence_length, input_size)
    state = model.initial_state(batch_size)
    for i in range(sequence_length):
        input_tensor = input_sequence[:, i]
        state = state.next(input_tensor)
        output = state.output()
        assert output.size() == (batch_size, input_size)
        # NOTE It's not possible in PyTorch to seed dropout so it's exactly
        # the same, so don't check that the outputs match.
