import torch

from .encoder import get_transformer_encoder
from .decoder import get_transformer_decoder

def get_transformer_encoder_decoder(
    source_vocabulary_size,
    target_input_vocabulary_size,
    target_output_vocabulary_size,
    tie_embeddings,
    num_encoder_layers,
    num_decoder_layers,
    d_model,
    num_heads,
    feedforward_size,
    dropout,
    use_source_padding=True,
    use_target_padding=True
):
    shared_embeddings = get_shared_embeddings(
        tie_embeddings,
        source_vocabulary_size,
        target_input_vocabulary_size,
        target_output_vocabulary_size,
        d_model,
        use_source_padding,
        use_target_padding
    )
    # NOTE It's ok to simply pass the same parameter to multiple sub-modules.
    # https://pytorch.org/docs/stable/notes/serialization.html#preserve-storage-sharing
    return TransformerEncoderDecoder(
        get_transformer_encoder(
            vocabulary_size=source_vocabulary_size,
            shared_embeddings=shared_embeddings,
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            feedforward_size=feedforward_size,
            dropout=dropout,
            use_padding=use_source_padding
        ),
        get_transformer_decoder(
            input_vocabulary_size=target_input_vocabulary_size,
            output_vocabulary_size=target_output_vocabulary_size,
            shared_embeddings=shared_embeddings,
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            feedforward_size=feedforward_size,
            dropout=dropout,
            use_padding=use_target_padding
        )
    )

def get_shared_embeddings(
    tie_embeddings,
    source_vocabulary_size,
    target_input_vocabulary_size,
    target_output_vocabulary_size,
    d_model,
    use_source_padding,
    use_target_padding
):
    if tie_embeddings:
        return construct_shared_embeddings(
            source_vocabulary_size,
            target_input_vocabulary_size,
            target_output_vocabulary_size,
            d_model,
            use_source_padding,
            use_target_padding
        )
    else:
        return None

def construct_shared_embeddings(
    source_vocabulary_size,
    target_input_vocabulary_size,
    target_output_vocabulary_size,
    d_model,
    use_source_padding,
    use_target_padding
):
    if target_output_vocabulary_size > target_input_vocabulary_size:
        raise ValueError
    vocab_size = max(
        source_vocabulary_size + int(use_source_padding),
        target_input_vocabulary_size + int(use_target_padding)
    )
    return torch.nn.Parameter(torch.zeros(vocab_size, d_model))

class TransformerEncoderDecoder(torch.nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source_sequence, target_sequence, source_is_padding_mask,
            target_is_padding_mask):
        encoder_outputs = self.encoder(
            source_sequence,
            is_padding_mask=source_is_padding_mask
        )
        return self.decoder(
            target_sequence,
            encoder_sequence=encoder_outputs,
            input_is_padding_mask=target_is_padding_mask,
            encoder_is_padding_mask=source_is_padding_mask,
            include_first=False
        )

    def initial_decoder_state(self, source_sequence, source_is_padding_mask):
        encoder_outputs = self.encoder(
            source_sequence,
            is_padding_mask=source_is_padding_mask
        )
        return self.decoder.initial_state(
            batch_size=encoder_outputs.size(0),
            encoder_sequence=encoder_outputs,
            encoder_is_padding_mask=source_is_padding_mask
        )
