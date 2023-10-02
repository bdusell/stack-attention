from collections.abc import Callable

import torch

class DifferentiableStack:

    def reading(self) -> torch.Tensor:
        raise NotImplementedError

    def transform_tensors(self,
        func: Callable[[torch.Tensor], torch.Tensor]
    ) -> 'DifferentiableStack':
        raise NotImplementedError

    def batch_size(self) -> int:
        raise NotImplementedError
