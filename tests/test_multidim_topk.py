import torch

from sequence_to_sequence.multidim_topk import multidim_topk

def test_2_dims():
    x = torch.tensor([
        [
            [3, 1, 9],
            [7, 2, 0],
            [4, 8, 5],
            [3, 5, 1]
        ],
        [
            [5, 3, 0],
            [1, 7, 2],
            [14, 5, 20],
            [7, 2, 9]
        ]
    ])
    expected_values = torch.tensor([
        [9, 8, 7],
        [20, 14, 9]
    ])
    expected_indexes = torch.tensor([
        [
            [0, 2],
            [2, 1],
            [1, 0]
        ],
        [
            [2, 2],
            [2, 0],
            [3, 2]
        ]
    ])
    topk_values, (topk_indexes_1, topk_indexes_2) = multidim_topk(x, k=3, dim=(1, 2), sorted=True)
    assert torch.all(topk_values == expected_values)
    assert torch.all(topk_indexes_1 == expected_indexes[:, :, 0])
    assert torch.all(topk_indexes_2 == expected_indexes[:, :, 1])
