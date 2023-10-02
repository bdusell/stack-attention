import unittest

import torch

from torch_unidirectional import UnidirectionalLSTM
from stack_rnn_models.grefenstette import GrefenstetteRNN

class TestGrefenstetteRNN(unittest.TestCase):

    def test_grefenstette(self):
        stack_embedding_size = 3
        input_size = 5
        hidden_units = 7
        batch_size = 11
        sequence_length = 13
        generator = torch.manual_seed(0)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = GrefenstetteRNN(
            input_size=input_size,
            stack_embedding_size=stack_embedding_size,
            controller=controller
        )
        for p in model.parameters():
            p.data.uniform_(generator=generator)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        input_tensor = torch.empty(batch_size, sequence_length - 1, input_size)
        input_tensor.uniform_(generator=generator)
        predicted_tensor = model(input_tensor)
        self.assertEqual(
            predicted_tensor.size(),
            (batch_size, sequence_length, hidden_units),
            'output has the expected dimensions'
        )
        target_tensor = torch.empty(batch_size, sequence_length, hidden_units)
        target_tensor.uniform_(generator=generator)
        loss = criterion(predicted_tensor, target_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    unittest.main()
