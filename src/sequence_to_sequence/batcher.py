import csv
import dataclasses
import math
import pathlib
from typing import Literal, Union

import humanfriendly
import torch

from .model_util import SequenceToSequenceModelInterface
from .memory_lookup import MemoryLookup
from .batching import (
    group_into_batches,
    group_sources_into_batches
)

def add_batching_arguments(group):
    group.add_argument('--batching-max-tokens', type=int)
    group.add_argument('--batching-space-coefficients', type=parse_coefficients)
    group.add_argument('--batching-precomputed-cost', type=pathlib.Path)
    group.add_argument('--batching-max-memory', type=humanfriendly.parse_size)

def parse_coefficients(s):
    return torch.tensor(list(map(float, s.split(','))))

def get_batcher(parser, args, model_interface):
    if (
        (args.batching_max_tokens is not None) +
        (args.batching_space_coefficients is not None) +
        (args.batching_precomputed_cost is not None)
    ) != 1:
        parser.error(
            'exactly one of --batching-max-tokens, '
            '--batching-space-coefficients, or '
            '--batching-precomputed-cost must be used'
        )
    if args.batching_max_tokens is not None:
        return MaxTokensBatcher(
            args.batching_max_tokens,
            model_interface
        )
    elif args.batching_space_coefficients is not None:
        if args.batching_max_memory is None:
            parser.error('--batching-max-memory is required')
        return PolynomialBatcher(
            args.batching_max_memory,
            model_interface,
            args.batching_space_coefficients
        )
    else:
        if args.batching_max_memory is None:
            parser.error('--batching-max-memory is required')
        return PrecomputedBatcher(
            args.batching_max_memory,
            model_interface,
            MemoryLookup(load_precomputed_memory(args.batching_precomputed_cost))
        )

@dataclasses.dataclass
class Batcher:

    max_cost: int
    model_interface: SequenceToSequenceModelInterface

    def filter_pairs(self, data):
        return filter_pairs(data, self.is_small_enough)

    def generate_batches(self, data):
        return group_into_batches(data, self.is_small_enough)

    def generate_source_batches(self, data):
        return group_sources_into_batches(data, self.is_small_enough)

    def is_small_enough(self, batch_size, source_length, target_length):
        estimated_cost = self.estimate_cost(
            batch_size,
            self.model_interface.adjust_source_length(source_length),
            self.model_interface.adjust_target_length(target_length)
        )
        return estimated_cost <= self.max_cost

    def estimate_cost(self, batch_size, source_length, target_length):
        return batch_size * self.estimate_cost_single(source_length, target_length)

    def estimate_cost_single(self, source_length, target_length):
        raise NotImplementedError

@dataclasses.dataclass
class MaxTokensBatcher(Batcher):

    def filter_pairs(self, data):
        return data

    def estimate_cost_single(self, source_length, target_length):
        return max(source_length, target_length)

@dataclasses.dataclass
class PolynomialBatcher(Batcher):

    coefficients: torch.Tensor

    def estimate_cost(self, batch_size, source_length, target_length):
        return evaluate_polynomial(
            self.coefficients,
            self.model_interface.get_space_polynomial_terms(batch_size, source_length, target_length)
        )

def evaluate_polynomial(coefficients: torch.tensor, terms: list[Union[int, float]]):
    return torch.inner(coefficients, coefficients.new_tensor(terms))

@dataclasses.dataclass
class PrecomputedBatcher(Batcher):

    lookup: MemoryLookup

    def estimate_cost_single(self, source_length, target_length):
        return self.lookup.lookup(source_length, target_length)

def load_precomputed_memory(filename):
    with filename.open() as fin:
        for row in csv.reader(fin, delimiter='\t', quoting=csv.QUOTE_NONE):
            if row[2] == 'OOM':
                continue
            source_length = int(row[0])
            target_length = int(row[1])
            memory_in_bytes = int(row[2])
            yield source_length, target_length, memory_in_bytes
