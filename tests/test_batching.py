import torch

from sequence_to_sequence.batching import (
    group_into_batches,
    group_sources_into_batches
)

def test_always_too_big():
    num_pairs = 17
    pairs = [
        (torch.tensor([i]), torch.tensor([i]))
        for i in range(num_pairs)
    ]
    batches = list(group_into_batches(pairs, lambda b, n, m: False))
    assert len(batches) == num_pairs
    batches = list(group_sources_into_batches((s for s, t in pairs), lambda b, n, m: False))
    assert len(batches) == num_pairs
