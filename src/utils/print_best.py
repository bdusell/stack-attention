import argparse
import math
import pathlib
import sys

import attr

from lib.pytorch_tools.saver import read_logs
from lib.logging import LogParseError
from cfl_language_modeling.lower_bound_perplexity import compute_cross_entropy_diff

@attr.s
class Trial:
    start = attr.ib()
    metrics = attr.ib()
    path = attr.ib()

def read_data_for_trial(dirname):
    start_data = None
    metrics = None
    try:
        with read_logs(dirname) as events:
            for event in events:
                if event.type == 'start':
                    start_data = event.data
                elif event.type == 'train':
                    src = event.data['best_validation_metrics']
                    metrics = { k : src[k] for k in ['perplexity'] }
    except (FileNotFoundError, LogParseError):
        pass
    if start_data is None or metrics is None:
        return None
    if 'valid_lower_bound_perplexity' in start_data:
        metrics['cross_entropy_diff'] = compute_cross_entropy_diff(
            metrics['perplexity'],
            start_data['valid_lower_bound_perplexity']
        )
    return Trial(start_data, metrics, dirname)

def read_data_for_multiple_trials(trial_dirs):
    trials = []
    missing_dirs = []
    for trial_dir in trial_dirs:
        trial = read_data_for_trial(trial_dir)
        if trial is not None:
            trials.append(trial)
        else:
            missing_dirs.append(trial_dir)
    return trials, missing_dirs

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='*', type=pathlib.Path)
    parser.add_argument('--metric', default='cross_entropy_diff')
    args = parser.parse_args()

    trials, missing_dirs = read_data_for_multiple_trials(args.inputs)
    for missing_dir in missing_dirs:
        print(f'missing: {missing_dir}', file=sys.stderr)
    if trials:
        best_trial = min(trials, key=lambda x: x.metrics[args.metric])
        best_trial_path = best_trial.path
    else:
        best_trial_path = ''
    print(f'{best_trial_path}\t{len(trials)}')

if __name__ == '__main__':
    main()
