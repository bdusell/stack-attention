import argparse
import dataclasses
import pathlib
import sys
from typing import Any, Callable, Optional

import humanfriendly

from lib.pytorch_tools.saver import read_logs
from lib.logging import LogParseError, LogEvent

@dataclasses.dataclass
class Trial:
    info: dict[str, dict[str, Any]]
    events: Optional[list[LogEvent]]
    path: pathlib.Path

def read_data_for_trial(dirname, capture_all_events):
    required_event_types = { 'model_info', 'start_training', 'train' }
    info = {}
    if capture_all_events:
        all_events = []
    else:
        all_events = None
    try:
        with read_logs(dirname) as events:
            for event in events:
                if event.type in required_event_types:
                    info[event.type] = event.data
                if capture_all_events:
                    all_events.append(event)
    except (FileNotFoundError, LogParseError):
        pass
    if len(info) != len(required_event_types):
        return None
    return Trial(info, all_events, dirname)

def read_data_for_multiple_trials(trial_dirs, capture_all_events):
    trials = []
    missing_dirs = []
    for trial_dir in trial_dirs:
        trial = read_data_for_trial(trial_dir, capture_all_events)
        if trial is not None:
            trials.append(trial)
        else:
            missing_dirs.append(trial_dir)
    return trials, missing_dirs

@dataclasses.dataclass
class Column:
    heading: str
    specifier: str
    key: str
    format: Callable
    bold_min: bool = False
    bold_max: bool = False

def format_text():
    return str

def format_float(places=2):
    def func(x):
        if x is not None:
            if isinstance(x, float):
                return f'{x:0.{places}f}'
            else:
                raise TypeError
        else:
            return ''
    return func

def format_int():
    def func(x):
        if x is not None:
            if isinstance(x, int):
                return humanfriendly.format_number(x)
            else:
                raise TypeError
        else:
            return ''
    return func

class Cache:

    def __init__(self, callbacks=None):
        super().__init__()
        self._cache = {}
        self._callbacks = {}
        if callbacks is not None:
            self.set_callbacks(callbacks)

    def __setitem__(self, key, value):
        self._cache[key] = value

    def set_callback(self, key, func):
        self._callbacks[key] = func

    def set_callbacks(self, callbacks):
        self._callbacks.update(callbacks)

    def __getitem__(self, key):
        if key in self._cache:
            return self._cache[key]
        elif key in self._callbacks:
            result = self[key] = self._callbacks[key](self)
            return result
        else:
            raise KeyError(f'unable to get item in cache for key {key!r}')

    def clear(self):
        self._cache = {}

def get_runs(cache):
    return len(cache['trials'])

def run_main(columns, callbacks, capture_all_events=False):

    parser = argparse.ArgumentParser()
    parser.add_argument('--label', action='append', default=[])
    parser.add_argument('--inputs', type=pathlib.Path, nargs='*', action='append', default=[])
    args = parser.parse_args()

    labels = args.label
    input_lists = args.inputs
    if len(labels) != len(input_lists):
        parser.error('must have the same number of --label and --input arguments')

    target_runs = max(len(input_list) for input_list in input_lists)
    labels_and_trials = []
    all_missing_dirs = []
    for label, input_list in zip(labels, input_lists):
        trials, missing_dirs = read_data_for_multiple_trials(input_list, capture_all_events)
        labels_and_trials.append((label, trials))
        all_missing_dirs.extend(missing_dirs)
    show_runs = not all(len(trials) == target_runs for label, trials in labels_and_trials)

    if show_runs:
        columns.append(Column('Runs', 'c', 'runs', format_int()))
        callbacks['runs'] = get_runs

    column_spec = ''.join(c.specifier for c in columns)
    print(f'\\begin{{tabular}}{{@{{}}{column_spec}@{{}}}}')
    print('\\toprule')

    print(' & '.join(c.heading for c in columns) + ' \\\\')
    print('\\midrule')

    caches = []
    for label, trials in labels_and_trials:
        cache = Cache(callbacks)
        cache['label'] = label
        cache['trials'] = trials
        caches.append(cache)

    min_values = {}
    max_values = {}
    for c in columns:
        if c.bold_min or c.bold_max:
            values = list(filter(lambda x: x is not None, (cache[c.key] for cache in caches)))
            if values:
                if c.bold_min:
                    min_values[c.key] = min(values)
                if c.bold_max:
                    max_values[c.key] = max(values)

    for cache in caches:
        cells = []
        for c in columns:
            value = cache[c.key]
            cell = c.format(value)
            if (
                (c.bold_min and c.key in min_values and value == min_values[c.key]) or
                (c.bold_max and c.key in max_values and value == max_values[c.key])
            ):
                cell = f'\\textbf{{{cell}}}'
            cells.append(cell)
        print(' & '.join(cells) + ' \\\\')

    print('\\bottomrule')
    print('\\end{tabular}')
    if show_runs:
        print(f'% info: results are not complete (targeting {target_runs} runs)')
    else:
        print(f'% info: all results are complete and are aggregated from {target_runs} runs')
    for missing_dir in all_missing_dirs:
        print(f'% missing: {missing_dir}', file=sys.stderr)
