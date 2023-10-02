import math

import torch

def make_causal_attention_mask(sequence_length, device, dtype):
    # TODO This can be cached, with slices taken from one big tensor.
    # Based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    # return : sequence_length x sequence_length
    # The return value will be added to the softmax logits for the attention
    # operation, so 0 leaves the weight alone, and -inf masks it out.
    # Dim 0 is the position being attended from, and dim 1 is the position in
    # the previous layer being attended to.
    # We want each position from to attend to everything except future
    # positions, so we want an upper-triangular matrix with -inf above the
    # diagonal and 0 everywhere else.
    neginf = torch.full((1, 1), -math.inf, device=device, dtype=dtype)
    neginf_square = neginf.expand(sequence_length, sequence_length)
    return torch.triu(neginf_square, diagonal=1)
