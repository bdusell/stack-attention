import torch

from transformer_model.input_layer import SinusoidalPositionalEncodingLayer
from transformer_model.decoder import get_transformer_decoder

def test_positional_encoding_forward_matches_iterative():
    batch_size = 3
    sequence_length = 5
    d_model = 32
    generator = torch.manual_seed(123)
    model = SinusoidalPositionalEncodingLayer()
    x = torch.empty(batch_size, sequence_length, d_model)
    x.uniform_(generator=generator)
    forward_output = model(x, include_first=False)
    state = model.initial_state(batch_size)
    for i in range(sequence_length):
        xi = x[:, i]
        state = state.next(xi)
        torch.testing.assert_close(state.output(), forward_output[:, i])

def test_forward_matches_iterative():
    batch_size = 5
    sequence_length = 13
    encoder_sequence_length = 17
    input_vocab_size = 7
    bos_index = input_vocab_size - 1
    output_vocab_size = 11
    d_model = 32
    generator = torch.manual_seed(123)
    model = get_transformer_decoder(
        input_vocabulary_size=input_vocab_size,
        output_vocabulary_size=output_vocab_size,
        shared_embeddings=None,
        positional_encoding_cacher=None,
        num_layers=1,
        d_model=d_model,
        num_heads=8,
        feedforward_size=64,
        dropout=0,
        use_padding=True
    )
    for param in model.parameters():
        param.data.uniform_(generator=generator)
    encoder_output = torch.empty(batch_size, encoder_sequence_length, d_model)
    encoder_output.uniform_(generator=generator)
    decoder_input = torch.concat([
        torch.full((batch_size, 1), bos_index, dtype=torch.long),
        torch.randint(input_vocab_size - 1, (batch_size, sequence_length), generator=generator)
    ], dim=1)
    input_is_padding_mask = torch.zeros(batch_size, sequence_length + 1)
    encoder_is_padding_mask = torch.zeros(batch_size, encoder_sequence_length)
    forward_output = model(
        decoder_input,
        encoder_output,
        input_is_padding_mask=input_is_padding_mask,
        encoder_is_padding_mask=encoder_is_padding_mask,
        include_first=False
    )
    assert forward_output.size() == (batch_size, sequence_length + 1, output_vocab_size)
    state = model.initial_state(batch_size, encoder_output, encoder_is_padding_mask)
    for i in range(sequence_length + 1):
        decoder_input_i = decoder_input[:, i]
        state = state.next(decoder_input_i)
        decoder_output_i = state.output()
        forward_output_i = forward_output[:, i]
        torch.testing.assert_close(decoder_output_i, forward_output_i, atol=1e-5, rtol=1e-5)
