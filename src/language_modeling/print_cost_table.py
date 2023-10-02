import humanfriendly
import numpy

from utils.print_table_util import run_main, Column, format_text, format_float, format_int

def format_bytes():
    return humanfriendly.format_size

def get_fastest_trial(cache):
    trials = cache['trials']
    if trials:
        return min(
            (
                (trial, get_trial_mean_epoch_duration(trial))
                for trial in trials
            ),
            key=lambda x: x[1]
        )
    else:
        return None

def get_trial_mean_epoch_duration(trial):
    return numpy.mean([
        event.data['duration']
        for event in trial.events
        if event.type == 'epoch'
    ])

def get_speed(cache):
    r = cache['fastest_trial']
    if r is None:
        return None
    trial, duration_per_epoch = r
    examples_per_epoch = trial.info['start_training']['training_examples']
    return examples_per_epoch / duration_per_epoch

def get_memory(cache):
    r = cache['fastest_trial']
    if r is None:
        return None
    trial, _ = r
    return max(
        event.data['peak_memory']
        for event in trial.events
        if event.type == 'epoch'
    )

def get_trial_no(cache):
    r = cache['fastest_trial']
    if r is None:
        return None
    trial, _ = r
    return trial.path

def main():
    run_main(
        columns=[
            Column('Model', 'l', 'label', format_text()),
            Column('Speed (examples/s)', 'c', 'speed', format_float(places=0)),
            Column('Memory', 'c', 'memory', format_bytes()),
            Column('Trial', 'c', 'trial_no', format_text())
        ],
        callbacks={
            'fastest_trial' : get_fastest_trial,
            'speed' : get_speed,
            'memory' : get_memory,
            'trial_no' : get_trial_no
        },
        capture_all_events=True
    )

if __name__ == '__main__':
    main()
