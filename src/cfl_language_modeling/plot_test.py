import argparse
import itertools
import math
import pathlib

import attr
import matplotlib.pyplot as plt
import tikzplotlib

from lib.pytorch_tools.saver import read_logs
from utils.plot_util import add_plot_arguments, run_plot, force_integer_ticks
from cfl_language_modeling.lower_bound_perplexity import compute_cross_entropy_diff

@attr.s
class Trial:
    lengths = attr.ib()
    metrics = attr.ib()

METRICS = ['perplexity']

def read_data_for_trial(dirname):
    lengths = []
    metrics = { k : [] for k in METRICS }
    metrics['cross_entropy_diff'] = []
    if dirname is not None:
        try:
            with read_logs(dirname) as events:
                for event in events:
                    if event.type == 'test_length':
                        lengths.append(event.data['length'])
                        for k in METRICS:
                            metrics[k].append(event.data[k])
                        perplexity = event.data['perplexity']
                        lower_bound_perplexity = event.data['lower_bound_perplexity']
                        metrics['cross_entropy_diff'].append(compute_cross_entropy_diff(perplexity, lower_bound_perplexity))
        except FileNotFoundError:
            pass
    return Trial(lengths, metrics)

def main():

    parser = argparse.ArgumentParser(
        description=
        'Generate plots of test set performance.'
    )
    parser.add_argument('--label', action='append', default=[])
    parser.add_argument('--input', type=pathlib.Path, nargs='?', action='append', default=[])
    parser.add_argument('--runs', type=int, action='append', default=[])
    parser.add_argument('--target-runs', type=int, required=True)
    parser.add_argument('--show-x-label', action='store_true', default=False)
    parser.add_argument('--vertical-line', type=int, default=80)
    add_plot_arguments(parser)
    args = parser.parse_args()

    num_models = len(args.input)
    if len(args.label) != num_models:
        parser.error('different number of --label flags than --input flags')
    if len(args.runs) != num_models:
        parser.error('different number of --run flags than --input flags')

    show_runs = any(runs != args.target_runs for runs in args.runs)

    with run_plot(args) as (fig, ax):
        ax.set_ylabel('Cross-entropy Diff.')
        if args.show_x_label:
            ax.set_xlabel('Length')
        ax.axvline(x=args.vertical_line, linestyle='--', color='black')
        for input_dir, label, runs in zip(args.input, args.label, args.runs):
            trial = read_data_for_trial(input_dir)
            if show_runs:
                label = f'{label} ({runs} runs)'
            x = trial.lengths
            y = trial.metrics['cross_entropy_diff']
            ax.plot(x, y, label=label)
        ax.set_ylim(bottom=0)
        force_integer_ticks(ax.xaxis)

if __name__ == '__main__':
    main()
