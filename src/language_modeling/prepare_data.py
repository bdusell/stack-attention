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
from language_modeling.vocabulary import build_softmax_vocab

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--training-files', type=pathlib.Path, nargs=2, required=True)
    parser.add_argument('--vocabulary-output', type=pathlib.Path, required=True)
    parser.add_argument('--more-files', type=pathlib.Path, nargs=2, action='append', default=[])
    parser.add_argument('--always-allow-unk', action='store_true', default=False)
    add_args(parser)
    parser.add_argument('--never-allow-unk', action='store_true', default=False)
    args = parser.parse_args()
    validate_args(parser, args)

    if args.always_allow_unk and args.never_allow_unk:
        parser.error('cannot pass both --always-allow-unk and --never-allow-unk')

    unk_string = None if args.never_allow_unk else args.unk_string

    token_types, has_unk = get_token_types_in_file(args.training_files[0], unk_string)
    allow_unk = (args.always_allow_unk or has_unk) and not args.never_allow_unk

    tokens = sorted(token_types)
    vocab = build_softmax_vocab(
        ToIntVocabularyBuilder(),
        tokens,
        allow_unk
    )

    print(f'token types: {len(token_types)}', file=sys.stderr)
    print(f'vocabulary size: {len(vocab)}', file=sys.stderr)
    print(f'has unk ({unk_string}): {has_unk}', file=sys.stderr)
    print(f'allow unk: {allow_unk}', file=sys.stderr)
    print(f'writing {args.vocabulary_output}', file=sys.stderr)
    torch.save({
        'tokens' : tokens,
        'allow_unk' : allow_unk
    }, args.vocabulary_output)
    for pair in [args.training_files] + args.more_files:
        prepare_file(vocab, pair)

if __name__ == '__main__':
    main()
