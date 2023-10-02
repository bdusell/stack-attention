from typing import Optional

import torch

from .layer import Layer

class TiedLinear(torch.nn.Module):

    def __init__(self, embeddings: torch.Tensor, output_size: int):
        r"""
        :param embeddings: A tensor of size :math:`V' \times I`, where
            :math:`V'` is at least ``output_size``, and :math:`I` is the size
            of the input vectors.
        """
        super().__init__()
        if embeddings.size(0) < output_size:
            raise ValueError(
                f'embeddings matrix only contains {embeddings.size(0)} '
                f'embeddings, but at least {output_size} are required'
            )
        self.embeddings = embeddings
        self._output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.embeddings[:self._output_size])

    def output_size(self) -> int:
        return self._output_size

def get_linear(
    input_size: int,
    output_size: int,
    shared_embeddings: Optional[torch.Tensor]=None,
    bias: bool=True
):
    if shared_embeddings is None:
        return Layer(input_size, output_size, bias=bias)
    else:
        if shared_embeddings.size(1) != input_size:
            raise ValueError(
                f'embeddings matrix has vectors of size {shared_embeddings.size(1)}, '
                f'but {input_size} was expected'
            )
        return TiedLinear(shared_embeddings, output_size)
