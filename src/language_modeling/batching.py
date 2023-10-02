from collections.abc import Callable, Iterable
from typing import Any

import torch

def group_into_batches(
    examples: list[torch.Tensor],
    is_small_enough: Callable[[int, int], bool]
) -> Iterable[list[tuple[torch.Tensor, torch.Tensor]]]:
    examples.sort(key=len)
    batch = []
    for example in examples:
        batch.append(example)
        batch_size = len(batch)
        max_length = len(example)
        # The initial batch always has size 1, and it should never be
        # discarded.
        # Since the sequences are sorted in increasing order of length, the
        # length of the current sequence is the maximum length in the batch.
        if (
            batch_size > 1 and
            not is_small_enough(batch_size, len(example))
        ):
            batch.pop()
            if batch:
                yield batch
                batch = [example]
    if batch:
        yield batch
