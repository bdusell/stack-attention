import dataclasses
import pathlib

import more_itertools
import torch

from vocab2 import ToStringVocabulary, ToStringVocabularyBuilder
from sequence_to_sequence.vocabulary import load_shared_vocabularies

@dataclasses.dataclass
class VocabularyContainer:
    source_vocab: ToStringVocabulary
    target_input_vocab: ToStringVocabulary
    target_output_vocab: ToStringVocabulary
    vocab_is_shared: bool

@dataclasses.dataclass
class Data(VocabularyContainer):
    training_data: list
    validation_data: list

def add_data_arguments(parser, validation=True):
    parser.add_argument('--training-data-source', type=pathlib.Path, required=True)
    parser.add_argument('--training-data-target', type=pathlib.Path, required=True)
    if validation:
        parser.add_argument('--validation-data-source', type=pathlib.Path, required=True)
        parser.add_argument('--validation-data-target', type=pathlib.Path, required=True)
    add_vocabulary_arguments(parser)

def add_vocabulary_arguments(parser):
    parser.add_argument('--shared-vocabulary', type=pathlib.Path)
    parser.add_argument('--source-vocabulary', type=pathlib.Path)
    parser.add_argument('--target-vocabulary', type=pathlib.Path)

def load_prepared_data(args, parser):
    training_data = load_prepared_data_files(args.training_data_source, args.training_data_target)
    if hasattr(args, 'validation_data_source'):
        validation_data = load_prepared_data_files(args.validation_data_source, args.validation_data_target)
    else:
        validation_data = None
    vocabs = load_vocabularies(args, parser)
    return Data(
        training_data=training_data,
        validation_data=validation_data,
        source_vocab=vocabs.source_vocab,
        target_input_vocab=vocabs.target_input_vocab,
        target_output_vocab=vocabs.target_output_vocab,
        vocab_is_shared=vocabs.vocab_is_shared
    )

def load_vocabularies(args, parser):
    if (
        (
            args.shared_vocabulary is not None and
            (args.source_vocabulary is not None or args.target_vocabulary is not None)
        ) or
        (
            args.shared_vocabulary is None and
            not (args.source_vocabulary is not None and args.target_vocabulary is not None)
        )
    ):
        parser.error('must provide either --shared-vocabulary or both --source-vocabulary and --target-vocabulary')
    if args.shared_vocabulary is not None:
        shared_vocabs = load_shared_vocabularies(args.shared_vocabulary, ToStringVocabularyBuilder())
        return VocabularyContainer(
            source_vocab=shared_vocabs.embedding_vocab,
            target_input_vocab=shared_vocabs.embedding_vocab,
            target_output_vocab=shared_vocabs.softmax_vocab,
            vocab_is_shared=True
        )
    else:
        raise NotImplementedError

def load_prepared_data_files(source_path, target_path):
    return list(more_itertools.zip_equal(
        load_prepared_data_file(source_path),
        load_prepared_data_file(target_path)
    ))

def load_prepared_data_file(path):
    return [torch.tensor(x) for x in torch.load(path)]
