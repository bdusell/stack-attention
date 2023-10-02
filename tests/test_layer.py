import unittest

import numpy
import torch

from torch_extras.layer import Layer, MultiLayer

class TestLayer(unittest.TestCase):

    def test_layer(self):
        batch_size = 5
        input_size = 7
        output_size = 11
        x = torch.ones(batch_size, input_size)
        layer = Layer(input_size, output_size, torch.nn.Softmax(dim=1))
        y = layer(x)
        self.assertEqual(y.size(), (batch_size, output_size))
        for a in y:
            a_sum = a.sum().item()
            self.assertAlmostEqual(a_sum, 1, places=6)
            for b in y:
                numpy.testing.assert_allclose(a.detach(), b.detach())

    def test_multi_layer(self):
        batch_size = 5
        input_size = 7
        output_size = 11
        n = 13
        x = torch.ones(batch_size, input_size)
        layer = MultiLayer(input_size, output_size, n, torch.nn.Softmax(dim=2))
        y = layer(x)
        self.assertEqual(y.size(), (batch_size, n, output_size))
        for a in y:
            for aa in a:
                aa_sum = aa.sum().item()
                self.assertAlmostEqual(aa_sum, 1, places=6)

if __name__ == '__main__':
    unittest.main()
