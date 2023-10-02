from collections.abc import Callable, Iterable
from typing import Any

import torch

def group_into_batches(
    pairs: list[tuple[torch.Tensor, torch.Tensor]],
    is_small_enough: Callable[[int, int, int], bool]
) -> Iterable[list[tuple[torch.Tensor, torch.Tensor]]]:
    pairs.sort(key=lambda x: len(x[0]) + len(x[1]))
    batch = []
    max_source_length = 0
    max_target_length = 0
    for example in pairs:
        batch.append(example)
        batch_size = len(batch)
        max_source_length = max(max_source_length, len(example[0]))
        max_target_length = max(max_target_length, len(example[1]))
        # The initial batch always has size 1, and it should never be
        # discarded.
        if (
            batch_size > 1 and
            not is_small_enough(batch_size, max_source_length, max_target_length)
        ):
            batch.pop()
            if batch:
                yield batch
                batch = [example]
                max_source_length = len(example[0])
                max_target_length = len(example[1])
    if batch:
        yield batch

def group_sources_into_batches(
    sources: Iterable[torch.Tensor],
    is_small_enough: Callable[[int, int, int], bool]
) -> Iterable[list[torch.Tensor]]:
    examples = sorted(enumerate(sources), key=lambda x: len(x[1]))
    batch = []
    max_source_length = 0
    for example in examples:
        batch.append(example)
        batch_size = len(batch)
        max_source_length = max(max_source_length, len(example[1]))
        if (
            batch_size > 1 and
            # Use source length to estimate target length.
            not is_small_enough(batch_size, max_source_length, max_source_length)
        ):
            batch.pop()
            if batch:
                yield batch
                batch = [example]
                max_source_length = len(example[1])
    if batch:
        yield batch
