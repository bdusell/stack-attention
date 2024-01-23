import dataclasses
import pathlib

from vocab2 import ToStringVocabulary, ToStringVocabularyBuilder
from sequence_to_sequence.data_util import load_prepared_data_file
from .vocabulary import load_shared_vocabularies

@dataclasses.dataclass
class VocabularyContainer:
    input_vocab: ToStringVocabulary
    output_vocab: ToStringVocabulary

@dataclasses.dataclass
class Data(VocabularyContainer):
    training_data: list
    validation_data: list

def add_data_arguments(parser, validation=True):
    parser.add_argument('--training-data', type=pathlib.Path, required=True)
    if validation:
        parser.add_argument('--validation-data', type=pathlib.Path, required=True)
    add_vocabulary_arguments(parser)

def add_vocabulary_arguments(parser):
    parser.add_argument('--vocabulary', type=pathlib.Path, required=True)

def load_prepared_data(args, parser):
    training_data = load_prepared_data_file(args.training_data)
    if hasattr(args, 'validation_data'):
        validation_data = load_prepared_data_file(args.validation_data)
    else:
        validation_data = None
    vocabs = load_vocabularies(args, parser)
    return Data(
        training_data=training_data,
        validation_data=validation_data,
        input_vocab=vocabs.input_vocab,
        output_vocab=vocabs.output_vocab
    )

def load_vocabularies(args, parser, builder=None):
    if builder is None:
        builder = ToStringVocabularyBuilder()
    vocabs = load_shared_vocabularies(args.vocabulary, builder)
    return VocabularyContainer(
        input_vocab=vocabs.embedding_vocab,
        output_vocab=vocabs.softmax_vocab,
    )
