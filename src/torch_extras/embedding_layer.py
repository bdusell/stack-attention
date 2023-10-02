from typing import Optional

import torch

class EmbeddingLayer(torch.nn.Module):

    def __init__(self,
        vocabulary_size: int,
        output_size: int,
        use_padding: bool,
        shared_embeddings: Optional[torch.Tensor]=None
    ):
        super().__init__()
        if use_padding:
            self.padding_index = vocabulary_size
            vocabulary_size += 1
        else:
            self.padding_index = None
        if shared_embeddings is not None:
            if shared_embeddings.size(0) < vocabulary_size:
                raise ValueError(
                    f'shared_embeddings has {shared_embeddings.size(0)} '
                    f'embeddings, but at least {vocabulary_size} are required'
                )
            if shared_embeddings.size(1) != output_size:
                raise ValueError(
                    f'shared_embeddings has embedding size '
                    f'{shared_embeddings.size(1)}, but it must be equal to '
                    f'{output_size}'
                )
            self.embeddings = shared_embeddings
        else:
            self.embeddings = torch.nn.Parameter(torch.zeros(
                vocabulary_size,
                output_size
            ))

    def forward(self, x):
        return torch.nn.functional.embedding(
            x,
            self.embeddings,
            padding_idx=self.padding_index
        )
