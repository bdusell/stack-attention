import json
import math

import numpy

from lib.pytorch_tools.saver import read_kwargs
from utils.print_table_util import run_main, Column, format_text, format_int, format_float

def get_best_trial(cache):
    trials = cache['trials']
    if trials:
        return trials[cache['best_index']]
    else:
        return None

def get_best_index(cache):
    trials = cache['trials']
    if trials:
        return numpy.argmin([
            trial.info['train']['best_validation_scores']['cross_entropy_per_token']
            for trial in trials
        ])
    else:
        return None

def get_kwargs(cache):
    best_trial = cache['best_trial']
    if best_trial:
        return read_kwargs(best_trial.path)
    else:
        return None

def get_params(cache):
    best_trial = cache['best_trial']
    if best_trial:
        return best_trial.info['model_info']['num_parameters']
    else:
        return None

def get_d_model(cache):
    kwargs = cache['kwargs']
    if kwargs is not None:
        return kwargs['d_model']
    else:
        return None

def get_val_perplexity(cache):
    best_trial = cache['best_trial']
    if best_trial:
        return math.exp(best_trial.info['train']['best_validation_scores']['cross_entropy_per_token'])
    else:
        return None

def read_test_data(path):
    scores_path = path / 'eval' / 'newstest2017-de-en' / 'scores.json'
    with scores_path.open() as fin:
        return json.load(fin)

def get_test_bleu(cache):
    best_trial = cache['best_trial']
    if best_trial:
        try:
            return read_test_data(best_trial.path)[0]['score']
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            return None
    else:
        return None

def main():
    run_main(
        columns=[
            Column('Model', 'l', 'label', format_text()),
            Column('\\dmodel', 'c', 'd_model', format_int()),
            Column('Params.', 'c', 'params', format_int()),
            Column('Val. Perp. $\\downarrow$', 'c', 'val_perplexity', format_float(places=2), bold_min=True),
            Column('Test BLEU $\\uparrow$', 'c', 'test_bleu', format_float(places=2), bold_max=True)
        ],
        callbacks={
            'best_trial' : get_best_trial,
            'best_index' : get_best_index,
            'kwargs' : get_kwargs,
            'params' : get_params,
            'd_model' : get_d_model,
            'val_perplexity' : get_val_perplexity,
            'test_bleu' : get_test_bleu
        }
    )

if __name__ == '__main__':
    main()
