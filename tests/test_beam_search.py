import dataclasses
import heapq

import more_itertools
import numpy
import torch

from torch_unidirectional import Unidirectional
from sequence_to_sequence.beam_search import beam_search

@dataclasses.dataclass
class CopyState(Unidirectional.State):

    correct_outputs: list[list[int]]
    is_valid: list[bool]
    timestep: int
    vocab_size: int

    def next(self, input_tensor):
        input_list = input_tensor.tolist()
        return dataclasses.replace(
            self,
            is_valid=[
                self.is_valid[i] and self.correct_outputs[i][self.timestep] == input_list[i]
                for i in range(len(self.correct_outputs))
            ],
            timestep=self.timestep + 1
        )

    def output(self):
        return torch.stack([
            self._get_batch_element_output(i)
            for i in range(self.batch_size())
        ], dim=0)

    def _get_batch_element_output(self, i):
        scores = torch.ones(self.vocab_size)
        if self.is_valid[i]:
            correct_output = self.correct_outputs[i][self.timestep]
            scores[correct_output] += 1
        return scores / torch.sum(scores)

    def batch_size(self):
        return len(self.correct_outputs)

    def transform_tensors(self, func):
        old_indexes = torch.arange(self.batch_size())
        new_indexes = func(old_indexes).tolist()
        return dataclasses.replace(
            self,
            correct_outputs=[self.correct_outputs[i] for i in new_indexes],
            is_valid=[self.is_valid[i] for i in new_indexes]
        )

def sequences_to_initial_copy_state(sequences, vocab_size):
    return CopyState(sequences, torch.tensor([True] * len(sequences)), 0, vocab_size)

def test_copy_state():
    correct_sequences = [
        [0, 1, 1, 2],
        [1, 0, 1, 2]
    ]
    state = sequences_to_initial_copy_state(correct_sequences, 3)
    for i in range(3):
        output = state.output()
        input = torch.tensor([
            correct_sequences[j][i]
            for j in range(len(correct_sequences))
        ])
        state = state.next(input)
    output = state.output()

def test_copy_beam_search():
    eos = 2
    correct_sequences = [
        [0, 1, 1],
        [],
        [1, 0, 1, 1, 0],
        [1],
        [0, 1],
        [1, 1, 1],
        [1, 1]
    ]
    state = sequences_to_initial_copy_state([s + [eos] for s in correct_sequences], 3)
    output_sequences = beam_search(
        initial_state=state,
        beam_size=5,
        eos_symbol=2,
        max_length=10,
        device=torch.device('cpu')
    )
    assert output_sequences == correct_sequences

@dataclasses.dataclass
class RandomState(Unidirectional.State):

    seeds: list[int]
    output_size: int

    def next(self, input_tensor):
        eos = self.output_size - 1
        if torch.any(input_tensor == eos).item():
            raise ValueError(f'input contains EOS: {input_tensor}')
        return dataclasses.replace(
            self,
            seeds=[
                abs(hash((seed, input)))
                for seed, input in more_itertools.zip_equal(self.seeds, input_tensor.tolist())
            ]
        )

    def output(self):
        return torch.stack([
            get_random_output(seed, self.output_size)
            for seed in self.seeds
        ], dim=0)

    def batch_size(self):
        return len(self.seeds)

    def transform_tensors(self, func):
        seeds_tensor = torch.tensor(self.seeds)
        new_seeds_tensor = func(seeds_tensor)
        return dataclasses.replace(self, seeds=new_seeds_tensor.tolist())

def get_random_output(seed, output_size):
    generator = numpy.random.Generator(numpy.random.PCG64(seed))
    numpy_values = generator.random((output_size,))
    torch_values = torch.from_numpy(numpy_values)
    scores = torch_values + 0.001
    return scores / torch.sum(scores)

def test_random_state_is_consistent():
    batch_size = 3
    sequence_length = 7
    output_size = 13
    input_sequence = torch.randint(output_size-1, (sequence_length, batch_size))
    initial_state = RandomState([0, 123, 42], output_size)
    state = initial_state
    assert state.batch_size() == 3
    outputs_1 = [state.output()]
    for input_tensor in input_sequence:
        state = state.next(input_tensor)
        outputs_1.append(state.output())
    permutation = torch.tensor([1, 2, 0])
    state = initial_state.transform_tensors(lambda x: x[permutation, ...])
    assert state.batch_size() == 3
    outputs_2 = [state.output()]
    for input_tensor in input_sequence:
        state = state.next(input_tensor[permutation, ...])
        outputs_2.append(state.output())
    for output_1, output_2 in more_itertools.zip_equal(outputs_1, outputs_2):
        assert torch.all(output_1[permutation, ...] == output_2)

def inefficient_beam_search(initial_state, beam_size, eos_symbol, max_length):
    return [
        inefficient_beam_search_single(
            initial_state.transform_tensors(lambda x: x[i:i+1, ...]),
            beam_size,
            eos_symbol,
            max_length
        )
        for i in range(initial_state.batch_size())
    ]

def inefficient_beam_search_single(initial_state, beam_size, eos_symbol, max_length):
    assert initial_state.batch_size() == 1
    beam = [
        dict(
            state=initial_state,
            log_probability=0.0,
            sequence=[],
            is_finished=False
        )
    ]
    for t in range(max_length):
        candidates = []
        # Split into two loops so that the ordering of ties is consistent with
        # the real implementation.
        for item in beam:
            if not item['is_finished']:
                output_logits, = item['state'].output()
                output_log_probs = torch.nn.functional.log_softmax(output_logits, dim=0)
                for symbol, output_log_prob in enumerate(output_log_probs.tolist()):
                    candidates.append(dict(
                        item=item,
                        log_probability=item['log_probability'] + output_log_prob,
                        # Should the "length" of candidates for non-EOS be
                        # considered 1 longer than the "length" of candidates
                        # for EOS? By "length" we really mean the number of
                        # predictions that the model has made, in which case
                        # they should be the same.
                        length=len(item['sequence']),
                        symbol=symbol
                    ))
        for item in beam:
            if item['is_finished']:
                candidates.append(dict(
                    item=item,
                    log_probability=item['log_probability'],
                    length=len(item['sequence']),
                    symbol=None
                ))
        top_candidates = heapq.nlargest(
            beam_size,
            candidates,
            # NOTE This implements length normalization. This includes EOS but
            # not BOS in the denominator. In other words, the denominator is
            # the number of symbols predicted by the model.
            key=lambda x: x['log_probability'] / (x['length'] + 1)
        )
        beam = []
        for candidate in top_candidates:
            item = candidate['item']
            if item['is_finished']:
                beam.append(item)
            else:
                symbol = candidate['symbol']
                if symbol == eos_symbol:
                    beam.append(dict(
                        state=None,
                        log_probability=candidate['log_probability'],
                        sequence=item['sequence'],
                        is_finished=True
                    ))
                else:
                    state = item['state']
                    input_tensor = torch.tensor([symbol])
                    beam.append(dict(
                        state=state.next(input_tensor),
                        log_probability=candidate['log_probability'],
                        sequence=item['sequence'] + [symbol],
                        is_finished=False
                    ))
        if beam[0]['is_finished']:
            break
    return beam[0]['sequence']

def test_beam_search_matches_reference():
    batch_size = 100
    vocab_size = 37
    eos = vocab_size - 1
    beam_size = 5
    max_length = 16
    initial_state = RandomState(list(range(batch_size)), vocab_size)
    expected_result = inefficient_beam_search(initial_state, beam_size, eos, max_length)
    # Make sure at least one of the sequences in this test case hits the max length.
    assert any(len(r) == max_length for r in expected_result)
    result = beam_search(initial_state, beam_size, eos, max_length, torch.device('cpu'))
    assert expected_result == result
