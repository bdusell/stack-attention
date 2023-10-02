import argparse
import logging
import pathlib
import random
import sys

import torch

from lib.lang_algorithm.pcfg import string_log_probability
from utils.cli_util import parse_interval
from cfl_language_modeling.lower_bound_perplexity import (
    compute_lower_bound_parts, parts_to_perplexity, Parts)
from cfl_language_modeling.data_util import generate_batches
from cfl_language_modeling.task_util import add_task_arguments, parse_task

def main():

    logger = logging.getLogger('main')
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description=
        'Generate a set of test data for a particular task, compute its '
        'lower bound perplexity, and save that information to a .pt file. '
        'Models can later be evaluated on the saved test set.'
    )
    parser.add_argument('--test-length-range', type=parse_interval, required=True,
        metavar='min:max',
        help='The range of lengths that the generated sequences will have.')
    parser.add_argument('--test-data-size', type=int, required=True,
        help='The number of sequences to generate per length.')
    parser.add_argument('--test-batch-size', type=int, default=256,
        help='Number of sequences per batch.')
    parser.add_argument('--test-data-seed', type=int,
        help='Random seed.')
    parser.add_argument('--output', type=pathlib.Path,
        help='Output file in .pt format.')
    add_task_arguments(parser)
    args = parser.parse_args()

    device = torch.device('cpu')
    task = parse_task(parser, args)
    sampler = task.sampler

    random_seed = args.test_data_seed
    if random_seed is None:
        random_seed = random.getrandbits(32)
    logger.info(f'random seed: {random_seed}')
    test_data_generator = random.Random(random_seed)
    test_perp_numer = 0.0
    test_acc_numer = 0
    test_num_symbols = 0
    test_num_samples = 0
    test_lower_perp_numer = 0.0
    valid_lengths = sampler.valid_lengths(*args.test_length_range)
    rows = []
    for length in valid_lengths:
        logger.info(f'length {length}')
        row = {}
        row['length'] = length
        batches = list(generate_batches(
            sampler=sampler,
            valid_lengths=[length],
            num_samples=args.test_data_size,
            batch_size=args.test_batch_size,
            generator=test_data_generator
        ))
        row['batches'] = batches
        parts = compute_lower_bound_parts(
            sampler=sampler,
            samples=(sequence for batch in batches for sequence in batch)
        )
        length_lower_perp = parts_to_perplexity(parts, 1)
        logger.info(f'  lower bound perplexity: {length_lower_perp:.3f}')
        row['lower_bound_perplexity_numerator'] = parts.total_neg_log_prob
        row['num_symbols'] = parts.total_len
        row['num_samples'] = parts.num_samples
        row['lower_bound_perplexity'] = length_lower_perp
        test_num_symbols += parts.total_len
        test_num_samples += parts.num_samples
        test_lower_perp_numer += parts.total_neg_log_prob
        rows.append(row)
    test_lower_perp = parts_to_perplexity(
        Parts(test_lower_perp_numer, test_num_symbols, test_num_samples),
        len(valid_lengths))
    logger.info(f'test lower bound perplexity: {test_lower_perp:.3f}')
    if args.output is not None:
        torch.save({
            'lengths' : rows,
            'total' : {
                'lower_bound_perplexity_numerator' : test_lower_perp_numer,
                'num_symbols' : test_num_symbols,
                'num_samples' : test_num_samples,
                'lower_bound_perplexity' : test_lower_perp
            },
            'input_vocab_size' : task.input_vocab.size(),
            'eos_symbol' : task.output_vocab.end_symbol
        }, args.output)

if __name__ == '__main__':
    main()
