import argparse
import logging
import math
import pathlib
import sys

import torch

from cfl_language_modeling.model_util import CFLModelInterface
from cfl_language_modeling.train_util import evaluate

def main():

    logger = logging.getLogger('main')
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    model_interface = CFLModelInterface(use_init=False, require_output=False)

    parser = argparse.ArgumentParser(
        description=
        'Load a set of test data and evaluate a model on it. Calculate '
        'performance on the whole test set and binned by string length.'
    )
    model_interface.add_arguments(parser)
    model_interface.add_forward_arguments(parser)
    parser.add_argument('--data', type=pathlib.Path, required=True,
        help='A .pt file containing a pre-generated test set.')
    parser.add_argument('--no-progress', action='store_true', default=False,
        help='Do not print progress messages.')
    args = parser.parse_args()

    device = model_interface.get_device(args)
    saver = model_interface.construct_saver(args)
    data = torch.load(args.data, map_location=device)

    with saver.logger() as events:
        test_perp_numer = 0.0
        test_acc_numer = 0
        input_vocab_size = data['input_vocab_size']
        eos_symbol = data['eos_symbol']
        for length_info in data['lengths']:
            length = length_info['length']
            batches = length_info['batches']
            logger.info(f'testing length {length}')
            result = evaluate(
                saver,
                model_interface.batches_to_tensors(batches, device, input_vocab_size, eos_symbol),
                model_interface,
                show_progress=not args.no_progress,
                logger=logger
            )
            # Get stats for just this length.
            length_perplexity = result['perplexity']
            length_lower_perp = length_info['lower_bound_perplexity']
            length_accuracy = result['accuracy']
            logger.info(f'  perplexity: {length_perplexity:.3f}')
            logger.info(f'  lower bound perplexity: {length_lower_perp:.3f}')
            logger.info(f'  accuracy:   {length_accuracy:.2%}')
            result['length'] = length
            result['lower_bound_perplexity'] = length_info['lower_bound_perplexity']
            events.log('test_length', result)
            # Aggregate stats for the whole test set.
            test_perp_numer += result['perplexity_numerator']
            test_acc_numer += result['accuracy_numerator']
        test_num_symbols = data['total']['num_symbols']
        test_perplexity = math.exp(test_perp_numer / test_num_symbols)
        test_accuracy = test_acc_numer / test_num_symbols
        test_lower_perp = data['total']['lower_bound_perplexity']
        logger.info(f'test perplexity: {test_perplexity:.3f}')
        logger.info(f'test lower bound perplexity: {test_lower_perp:.3f}')
        logger.info(f'test accuracy:   {test_accuracy:.2%}')
        events.log('test', {
            'perplexity' : test_perplexity,
            'accuracy' : test_accuracy,
            'lower_bound_perplexity' : test_lower_perp
        })

if __name__ == '__main__':
    main()
