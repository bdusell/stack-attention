import torch

from torch_unidirectional import SimpleUnidirectional, SimpleLayerUnidirectional

class MySimple(SimpleUnidirectional):

    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

    def forward_single(self, input_tensor):
        c = torch.sum(input_tensor, dim=1, keepdim=True)
        return input_tensor + c

    def forward_sequence(self, input_sequence):
        c = torch.sum(input_sequence, dim=2, keepdim=True)
        return input_sequence + c

    def initial_output(self, batch_size):
        return torch.ones((batch_size, self.input_size))

def test_forward_matches_iterative():
    batch_size = 5
    sequence_length = 13
    input_size = 7
    model = MySimple(input_size)
    generator = torch.manual_seed(123)
    input_sequence = torch.rand((batch_size, sequence_length, input_size), generator=generator)
    forward_output = model(input_sequence, include_first=False)
    assert forward_output.size() == (batch_size, sequence_length, input_size)
    state = model.initial_state(batch_size)
    for i in range(sequence_length):
        input_tensor = input_sequence[:, i]
        state = state.next(input_tensor)
        output = state.output()
        assert output.size() == (batch_size, input_size)
        torch.testing.assert_close(output, forward_output[:, i])

def test_custom_args():
    batch_size = 5
    sequence_length = 13
    input_size = 7
    def func(x, alpha, *, beta):
        return x * alpha + beta
    model = SimpleLayerUnidirectional(func)
    generator = torch.manual_seed(123)
    input_sequence = torch.rand((batch_size, sequence_length, input_size), generator=generator)
    forward_output = model(input_sequence, 123.0, include_first=False, beta=456.0)
    assert forward_output.size() == (batch_size, sequence_length, input_size)
    state = model.initial_state(batch_size, 123.0, beta=456.0)
    for i in range(sequence_length):
        input_tensor = input_sequence[:, i]
        state = state.next(input_tensor)
        output = state.output()
        assert output.size() == (batch_size, input_size)
        torch.testing.assert_close(output, forward_output[:, i])
