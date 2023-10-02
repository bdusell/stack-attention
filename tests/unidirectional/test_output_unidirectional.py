import torch

from torch_unidirectional import OutputUnidirectional

def test_forward_and_iterative():
    batch_size = 5
    sequence_length = 13
    input_size = 7
    vocabulary_size = 17
    generator = torch.manual_seed(123)
    shared_embeddings = torch.nn.Parameter(torch.zeros((vocabulary_size + 20, input_size)))
    model = OutputUnidirectional(
        input_size=input_size,
        vocabulary_size=vocabulary_size,
        shared_embeddings=shared_embeddings
    )
    for p in model.parameters():
        p.data.uniform_(generator=generator)
    input_sequence = torch.rand((batch_size, sequence_length, input_size), generator=generator)
    forward_output = model(input_sequence, include_first=False)
    assert forward_output.size() == (batch_size, sequence_length, vocabulary_size)
    state = model.initial_state(batch_size)
    for i in range(sequence_length):
        input_tensor = input_sequence[:, i]
        state = state.next(input_tensor)
        output = state.output()
        assert output.size() == (batch_size, vocabulary_size)
        torch.testing.assert_close(output, forward_output[:, i])
