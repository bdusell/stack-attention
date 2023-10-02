import argparse
import pathlib
import sys

import torch

from vocab2 import ToIntVocabularyBuilder
from sequence_to_sequence.prepare_data_utils import (
    add_args,
    validate_args,
    get_token_types_in_file,
    prepare_file
)
from sequence_to_sequence.vocabulary import build_shared_vocabularies

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--source-training-files', type=pathlib.Path, nargs=2, required=True)
    parser.add_argument('--target-training-files', type=pathlib.Path, nargs=2, required=True)
    parser.add_argument('--vocabulary-output', type=pathlib.Path, required=True)
    parser.add_argument('--more-source-files', type=pathlib.Path, nargs=2, action='append', default=[])
    parser.add_argument('--more-target-files', type=pathlib.Path, nargs=2, action='append', default=[])
    parser.add_argument('--always-allow-unk', action='store_true', default=False)
    add_args(parser)
    args = parser.parse_args()
    validate_args(parser, args)

    source_token_types, source_has_unk = get_token_types_in_file(args.source_training_files[0], args.unk_string)
    target_token_types, target_has_unk = get_token_types_in_file(args.target_training_files[0], args.unk_string)
    allow_unk = args.always_allow_unk or source_has_unk or target_has_unk

    tokens_in_target = sorted(target_token_types)
    tokens_only_in_source = sorted(source_token_types - target_token_types)
    vocabs = build_shared_vocabularies(
        ToIntVocabularyBuilder(),
        tokens_in_target,
        tokens_only_in_source,
        allow_unk
    )

    print(f'token types in target: {len(tokens_in_target)}', file=sys.stderr)
    print(f'token types only in source: {len(tokens_only_in_source)}', file=sys.stderr)
    print(f'embedding vocabulary size: {len(vocabs.embedding_vocab)}', file=sys.stderr)
    print(f'softmax vocabulary size: {len(vocabs.softmax_vocab)}', file=sys.stderr)
    print(f'source has {args.unk_string}: {source_has_unk}', file=sys.stderr)
    print(f'target has {args.unk_string}: {target_has_unk}', file=sys.stderr)
    print(f'allow unk: {allow_unk}', file=sys.stderr)
    print(f'writing {args.vocabulary_output}', file=sys.stderr)
    torch.save({
        'tokens_in_target' : tokens_in_target,
        'tokens_only_in_source' : tokens_only_in_source,
        'allow_unk' : allow_unk
    }, args.vocabulary_output)
    for pair in [args.source_training_files] + args.more_source_files:
        prepare_file(vocabs.embedding_vocab, pair)
    for pair in [args.target_training_files] + args.more_target_files:
        prepare_file(vocabs.softmax_vocab, pair)

if __name__ == '__main__':
    main()
