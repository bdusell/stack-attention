import math

import torch

def sinusoidal_positional_encodings(sequence_length, d_model, device):
    # Based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    if sequence_length == 0 or d_model == 0:
        return torch.empty((sequence_length, d_model), device=device)
    # TODO This doesn't work when d_model is odd.
    position = torch.arange(sequence_length, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device) *
        (-math.log(10000.0) / d_model)
    )
    pe = torch.empty(sequence_length, d_model, device=device)
    # TODO I'm sure sin and cos can be parallelized by simply changing the
    # phase of sin.
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class SinusoidalPositionalEncodingCacher(torch.nn.Module):
    """A module that caches a tensor of sinusoidal positional encodings.

    Note that it is highly recommended to set a maximum size up-front before
    training to avoid CUDA memory fragmentation.
    """

    def __init__(self):
        super().__init__()
        self._set_cache_size_with_device((0, 0), None)
        self._allow_reallocation = True

    def clear(self):
        self._set_cache_size((0, 0))

    def _set_cache_size(self, size):
        self._set_cache_size_with_device(size, self.encodings.device)

    def _set_cache_size_with_device(self, size, device):
        sequence_length, d_model = size
        self._set_encodings(sinusoidal_positional_encodings(
            sequence_length,
            d_model,
            device
        ))

    def _set_encodings(self, tensor):
        self.register_buffer('encodings', tensor, persistent=False)

    def get_encodings(self, sequence_length, d_model):
        query_size = (sequence_length, d_model)
        cache_size = self.encodings.size()
        if not all(a <= b for a, b in zip(query_size, cache_size)):
            if not self._allow_reallocation:
                raise ValueError(
                    'reallocation of the positional encoding cache has been '
                    'intentionally disabled with set_allow_reallocation(False)'
                )
            # Make sure never to decrease the cached sequence_length or
            # d_model to avoid flip-flopping.
            new_size = tuple(max(a, b) for a, b in zip(query_size, cache_size))
            self._set_cache_size(new_size)
        return self.encodings[:sequence_length, :d_model]

    def set_allow_reallocation(self, value):
        self._allow_reallocation = value
