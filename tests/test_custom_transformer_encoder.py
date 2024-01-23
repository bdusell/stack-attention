import random

import torch

from torch_extras.init import smart_init, uniform_fallback
from vocab2 import ToStringVocabularyBuilder, ToIntVocabularyBuilder
from sequence_to_sequence.prepare_data_utils import get_token_types
from sequence_to_sequence.vocabulary import build_shared_vocabularies
from sequence_to_sequence.data_util import VocabularyContainer
from sequence_to_sequence.model_util import SequenceToSequenceModelInterface

def test_custom_matches_builtin():

    batch_size = 7

    generator = random.Random(123)
    def random_sequence():
        length = generator.randint(10, 20)
        return [
            chr(generator.randint(ord('a'), ord('z')))
            for i in range(length)
        ]

    batch = [(random_sequence(), random_sequence()) for i in range(batch_size)]
    source_token_types, source_has_unk = get_token_types((c for x, y in batch for c in x), '<unk>')
    target_token_types, target_has_unk = get_token_types((c for x, y in batch for c in y), '<unk>')
    tokens_in_target = sorted(target_token_types)
    tokens_only_in_source = sorted(source_token_types - target_token_types)
    shared_vocabs_to_int = build_shared_vocabularies(
        ToIntVocabularyBuilder(),
        tokens_in_target,
        tokens_only_in_source,
        allow_unk=False
    )
    batch_as_ints = [
        (
            torch.tensor([shared_vocabs_to_int.embedding_vocab.to_int(c) for c in x]),
            torch.tensor([shared_vocabs_to_int.softmax_vocab.to_int(c) for c in y])
        )
        for x, y in batch
    ]
    shared_vocabs = build_shared_vocabularies(
        ToStringVocabularyBuilder(),
        tokens_in_target,
        tokens_only_in_source,
        allow_unk=False
    )
    vocabs = VocabularyContainer(
        source_vocab=shared_vocabs.embedding_vocab,
        target_input_vocab=shared_vocabs.embedding_vocab,
        target_output_vocab=shared_vocabs.softmax_vocab,
        vocab_is_shared=True
    )

    source_vocab_size = 11
    target_vocab_size = 13
    model_interface = SequenceToSequenceModelInterface(
        use_load=False,
        use_init=False,
        use_output=False,
        require_output=False
    )
    model_interface.block_size = 10
    def construct_model(use_standard_module):
        return model_interface.construct_model(
            encoder_layers='3',
            use_standard_encoder=use_standard_module,
            decoder_layers='3',
            use_standard_decoder=use_standard_module,
            d_model=32,
            num_heads=4,
            feedforward_size=128,
            dropout=0.2,
            source_vocab_size=len(vocabs.source_vocab),
            target_input_vocab_size=len(vocabs.target_input_vocab),
            target_output_vocab_size=len(vocabs.target_output_vocab),
            tie_embeddings=True
        )
    builtin_model = construct_model(True)
    custom_model = construct_model(False)
    def construct_generator():
        return torch.manual_seed(123)
    def init_model(model, generator):
        smart_init(model, generator, fallback=uniform_fallback(0.1))

    generator = construct_generator()
    init_model(builtin_model, generator)

    generator = construct_generator()
    init_model(custom_model, generator)

    model_input, target_output = model_interface.prepare_batch(batch_as_ints, torch.device('cpu'), vocabs)

    torch.manual_seed(42)
    builtin_logits = model_interface.get_logits(builtin_model, model_input)
    torch.manual_seed(42)
    custom_logits = model_interface.get_logits(custom_model, model_input)
    torch.testing.assert_close(builtin_logits, custom_logits)
