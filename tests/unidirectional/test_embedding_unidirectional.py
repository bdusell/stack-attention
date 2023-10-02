import torch

from torch_unidirectional import EmbeddingUnidirectional

def test_forward_and_iterative():
    batch_size = 5
    sequence_length = 13
    output_size = 7
    vocabulary_size = 17
    generator = torch.manual_seed(123)
    model = EmbeddingUnidirectional(
        vocabulary_size=vocabulary_size,
        output_size=output_size,
        use_padding=True
    )
    for p in model.parameters():
        p.data.uniform_(generator=generator)
    input_sequence = torch.randint(vocabulary_size, (batch_size, sequence_length), generator=generator)
    forward_output = model(input_sequence, include_first=False)
    assert forward_output.size() == (batch_size, sequence_length, output_size)
    state = model.initial_state(batch_size)
    for i in range(sequence_length):
        input_tensor = input_sequence[:, i]
        state = state.next(input_tensor)
        output = state.output()
        assert output.size() == (batch_size, output_size)
        torch.testing.assert_close(output, forward_output[:, i])
