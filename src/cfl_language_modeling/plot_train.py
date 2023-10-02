import argparse
import math
import pathlib
import sys

import attr
import matplotlib.pyplot as plt
import tikzplotlib

from lib.pytorch_tools.saver import read_logs
from utils.plot_util import add_plot_arguments, run_plot, force_integer_ticks
from cfl_language_modeling.lower_bound_perplexity import compute_cross_entropy_diff

@attr.s
class Trial:
    checkpoints = attr.ib()

METRICS = ['perplexity']

def read_data_for_trial(dirname):
    lower_bound_perplexity = None
    checkpoints = { k : [] for k in METRICS }
    if dirname is not None:
        try:
            with read_logs(dirname) as events:
                for event in events:
                    if event.type == 'start':
                        lower_bound_perplexity = event.data['valid_lower_bound_perplexity']
                    elif event.type == 'validate':
                        for k in METRICS:
                            checkpoints[k].append(event.data[k])
        except FileNotFoundError:
            pass
    checkpoints['cross_entropy_diff'] = [
        compute_cross_entropy_diff(perplexity, lower_bound_perplexity)
        for perplexity in checkpoints['perplexity']
    ]
    return Trial(checkpoints)

def main():

    parser = argparse.ArgumentParser(
        description=
        'Generate plots of validation set performance during training.'
    )
    parser.add_argument('--label', action='append', default=[])
    parser.add_argument('--input', type=pathlib.Path, nargs='?', action='append', default=[])
    parser.add_argument('--runs', type=int, action='append', default=[])
    parser.add_argument('--target-runs', type=int, required=True)
    parser.add_argument('--show-x-label', action='store_true', default=False)
    add_plot_arguments(parser)
    args = parser.parse_args()

    num_models = len(args.input)
    if len(args.label) != num_models:
        parser.error('different number of --label flags than --input flags')
    if len(args.runs) != num_models:
        parser.error('different number of --runs flags than --input flags')

    show_runs = any(runs != args.target_runs for runs in args.runs)

    with run_plot(args) as (fig, ax):
        ax.set_ylabel('Cross-entropy Diff.')
        if args.show_x_label:
            ax.set_xlabel('Epoch')
        for input_dir, label, runs in zip(args.input, args.label, args.runs):
            trial = read_data_for_trial(input_dir)
            if show_runs:
                label = f'{label} ({runs} runs)'
            y = trial.checkpoints['cross_entropy_diff']
            x = range(1, len(y)+1)
            ax.plot(x, y, label=label)
        ax.set_ylim(bottom=0)
        force_integer_ticks(ax.xaxis)

if __name__ == '__main__':
    main()
