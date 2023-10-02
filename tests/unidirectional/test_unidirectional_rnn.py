import pytest
import torch

from torch_unidirectional import UnidirectionalSimpleRNN, UnidirectionalLSTM
from torch_unidirectional.rnn import remove_extra_bias_parameters

@pytest.mark.parametrize('ModelClass', [UnidirectionalSimpleRNN, UnidirectionalLSTM])
@pytest.mark.parametrize('use_extra_bias', [True, False])
def test_forward_matches_iterative(ModelClass, use_extra_bias):
    batch_size = 5
    sequence_length = 13
    input_size = 7
    hidden_units = 17
    generator = torch.manual_seed(123)
    model = ModelClass(
        input_size=input_size,
        hidden_units=hidden_units,
        layers=3,
        use_extra_bias=use_extra_bias
    )
    for p in model.parameters():
        p.data.uniform_(generator=generator)
    input_sequence = torch.rand((batch_size, sequence_length, input_size), generator=generator)
    forward_output = model(input_sequence, include_first=True)
    assert forward_output.size() == (batch_size, sequence_length + 1, hidden_units)
    state = model.initial_state(batch_size)
    output = state.output()
    assert output.size() == (batch_size, hidden_units)
    torch.testing.assert_close(output, forward_output[:, 0])
    for i in range(sequence_length):
        input_tensor = input_sequence[:, i]
        state = state.next(input_tensor)
        output = state.output()
        assert output.size() == (batch_size, hidden_units)
        torch.testing.assert_close(output, forward_output[:, i+1])
    # Run a backward pass and make sure there are no NaNs.
    loss = torch.sum(forward_output)
    assert is_finite(loss)
    loss.backward()
    for param in model.parameters():
        assert is_finite(param.grad)

@pytest.mark.parametrize('ModelClass', [UnidirectionalSimpleRNN, UnidirectionalLSTM])
def test_no_extra_bias_differs_from_with_extra_bias(ModelClass):
    batch_size = 5
    sequence_length = 13
    input_size = 7
    hidden_units = 17
    generator = torch.manual_seed(123)
    model_with_extra_bias = ModelClass(
        input_size=input_size,
        hidden_units=hidden_units,
        layers=3,
        use_extra_bias=True
    )
    model_no_extra_bias = ModelClass(
        input_size=input_size,
        hidden_units=hidden_units,
        layers=3,
        use_extra_bias=False
    )
    assert get_total_params(model_no_extra_bias) < get_total_params(model_with_extra_bias)
    no_extra_bias_params = dict(model_no_extra_bias.named_parameters())
    for name, param in model_with_extra_bias.named_parameters():
        param.data.uniform_(generator=generator)
        # Make sure the parameter values are the same between the two models.
        if name in no_extra_bias_params:
            no_extra_bias_params[name].data.copy_(param.data)
    input_sequence = torch.rand((batch_size, sequence_length, input_size), generator=generator)
    forward_output_with_extra_bias = model_with_extra_bias(input_sequence, include_first=True)
    forward_output_no_extra_bias = model_no_extra_bias(input_sequence, include_first=True)
    assert_not_equal(forward_output_with_extra_bias, forward_output_no_extra_bias)
    loss_with_extra_bias = torch.sum(forward_output_with_extra_bias)
    loss_no_extra_bias = torch.sum(forward_output_no_extra_bias)
    assert is_finite(loss_with_extra_bias)
    assert is_finite(loss_no_extra_bias)
    loss_with_extra_bias.backward()
    loss_no_extra_bias.backward()
    for model in (model_with_extra_bias, model_no_extra_bias):
        for param in model.parameters():
            assert is_finite(param.grad)
    with_extra_bias_params = dict(model_with_extra_bias.named_parameters())
    for name, param in model_no_extra_bias.named_parameters():
        if 'bias' in name:
            assert_not_equal(param.grad, with_extra_bias_params[name].grad, f'parameter {name}')

@pytest.mark.parametrize('ModelClass', [UnidirectionalSimpleRNN, UnidirectionalLSTM])
def test_remove_extra_bias(ModelClass):
    batch_size = 5
    sequence_length = 13
    input_size = 7
    hidden_units = 17
    generator = torch.manual_seed(123)
    model = torch.nn.RNN(
        input_size,
        hidden_units,
        num_layers=3,
        batch_first=True,
        bidirectional=False
    )
    extra_bias_params = [
        param
        for name, param in model.named_parameters()
        if name.startswith('bias_hh_l')
    ]
    for param in model.parameters():
        param.data.uniform_(generator=generator)
    input_sequence = torch.rand((batch_size, sequence_length, input_size), generator=generator)
    forward_output, _ = model(input_sequence)
    num_params_before = get_total_params(model)
    remove_extra_bias_parameters(model)
    num_params_after = get_total_params(model)
    assert num_params_after < num_params_before
    forward_output_no_extra_bias, _ = model(input_sequence)
    assert_not_equal(forward_output, forward_output_no_extra_bias)
    loss = torch.sum(forward_output_no_extra_bias)
    loss.backward()
    for param in extra_bias_params:
        assert param.grad is None

def is_finite(tensor):
    return torch.all(torch.isfinite(tensor)).item()

def get_total_params(module):
    return sum(p.numel() for p in module.parameters())

def assert_not_equal(a, b, msg=None):
    assert not torch.equal(a, b), msg
