import torch

from transformer_model.unidirectional_encoder import UnidirectionalTransformerEncoderLayers

def test_layers_forward_matches_iterative():
    batch_size = 5
    sequence_length = 13
    d_model = 32
    generator = torch.manual_seed(123)
    model = UnidirectionalTransformerEncoderLayers(
        num_layers=3,
        d_model=d_model,
        num_heads=8,
        feedforward_size=64,
        dropout=0,
        use_final_layer_norm=True,
        enable_nested_tensor=False
    )
    for param in model.parameters():
        param.data.uniform_(generator=generator)
    input_sequence = torch.rand((batch_size, sequence_length, d_model), generator=generator)
    forward_output = model(input_sequence, include_first=False)
    assert forward_output.size() == (batch_size, sequence_length, d_model)
    state = model.initial_state(batch_size)
    for i in range(sequence_length):
        input_tensor = input_sequence[:, i]
        state = state.next(input_tensor)
        output = state.output()
        assert output.size() == (batch_size, d_model)
        torch.testing.assert_close(output, forward_output[:, i])
