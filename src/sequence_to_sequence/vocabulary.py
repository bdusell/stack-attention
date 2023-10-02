import dataclasses

import torch

from vocab2 import Vocabulary

def build_softmax_vocab(builder, tokens_in_target, allow_unk):
    result = builder.content(tokens_in_target)
    if allow_unk:
        result = result + builder.catchall('unk')
    return result + builder.reserved(['eos'])

def build_embedding_vocab(builder, softmax_vocab, tokens_only_in_source):
    return (
        softmax_vocab +
        builder.content(tokens_only_in_source) +
        builder.reserved(['bos'])
    )

@dataclasses.dataclass
class SharedVocabularies:
    softmax_vocab: Vocabulary
    embedding_vocab: Vocabulary

def build_shared_vocabularies(builder, tokens_in_target, tokens_only_in_source, allow_unk):
    softmax_vocab = build_softmax_vocab(builder, tokens_in_target, allow_unk)
    embedding_vocab = build_embedding_vocab(builder, softmax_vocab, tokens_only_in_source)
    return SharedVocabularies(softmax_vocab, embedding_vocab)

def load_shared_vocabularies(path, builder):
    data = torch.load(path)
    return build_shared_vocabularies(
        builder,
        data['tokens_in_target'],
        data['tokens_only_in_source'],
        data['allow_unk']
    )
