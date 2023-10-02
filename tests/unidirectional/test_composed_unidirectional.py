import torch

from torch_unidirectional import (
    ComposedUnidirectional,
    PositionalUnidirectional,
    SimpleLayerUnidirectional
)

class AdditivePositional(PositionalUnidirectional):

    def forward_from_position(self, input_sequence, position):
        indexes = torch.arange(
            position,
            position + input_sequence.size(1),
            device=input_sequence.device
        )
        return input_sequence + indexes[None]

    def forward_at_position(self, input_tensor, position):
        return input_tensor + position

class MultiplicativePositional(PositionalUnidirectional):

    def forward_from_position(self, input_sequence, position):
        indexes = torch.arange(
            position,
            position + input_sequence.size(1),
            device=input_sequence.device
        )
        return input_sequence * indexes[None]

    def forward_at_position(self, input_tensor, position):
        return input_tensor * position

def test_forward_matches_iterative():
    first_model = AdditivePositional()
    second_model = MultiplicativePositional()
    model = ComposedUnidirectional(first_model, second_model)
    batch_size = 5
    sequence_length = 13
    generator = torch.manual_seed(123)
    input_sequence = torch.rand((batch_size, sequence_length), generator=generator)
    expected_forward_output = second_model(
        first_model(
            input_sequence,
            include_first=False
        ),
        include_first=False
    )
    assert expected_forward_output.size() == (batch_size, sequence_length)
    forward_output = model(input_sequence, include_first=False)
    assert forward_output.size() == (batch_size, sequence_length)
    torch.testing.assert_close(forward_output, expected_forward_output)
    state = model.initial_state(batch_size)
    for i in range(sequence_length):
        input_tensor = input_sequence[:, i]
        state = state.next(input_tensor)
        output = state.output()
        assert output.size() == (batch_size,)
        torch.testing.assert_close(output, expected_forward_output[:, i])

class MainUnidirectional(SimpleLayerUnidirectional):

    def __init__(self):
        super().__init__(torch.nn.Identity())

    def initial_state(self, batch_size, x, y, alpha, beta):
        assert x == 'foo'
        assert y == 42
        assert alpha == 123
        assert beta == 'asdf'
        return super().initial_state(batch_size)

def test_arg_routing_to_main():
    model = (
        AdditivePositional() |
        MainUnidirectional().main() |
        MultiplicativePositional()
    )
    batch_size = 5
    sequence_length = 13
    generator = torch.manual_seed(123)
    input_sequence = torch.rand((batch_size, sequence_length), generator=generator)
    x = 'foo'
    y = 42
    alpha = 123
    beta = 'asdf'
    state = model.initial_state(batch_size, x, y, alpha=alpha, beta=beta)
    output = model(input_sequence, x, y, alpha=alpha, beta=beta, include_first=False)

class OtherUnidirectional(SimpleLayerUnidirectional):

    def __init__(self):
        super().__init__(torch.nn.Identity())

    def initial_state(self, batch_size, beta, gamma):
        assert beta == 999
        assert gamma == 'qwerty'
        return super().initial_state(batch_size)

class YetAnotherUnidirectional(SimpleLayerUnidirectional):

    def __init__(self):
        super().__init__(torch.nn.Identity())

    def initial_state(self, batch_size, alpha, delta):
        assert alpha == 'meow'
        assert delta == 'moo'
        return super().initial_state(batch_size)

def test_arg_routing_to_tags():
    model = (
        AdditivePositional() |
        MainUnidirectional().main() |
        AdditivePositional() |
        OtherUnidirectional().tag('other') |
        MultiplicativePositional() |
        YetAnotherUnidirectional().tag('yetanother')
    )
    batch_size = 5
    sequence_length = 13
    generator = torch.manual_seed(123)
    input_sequence = torch.rand((batch_size, sequence_length), generator=generator)
    x = 'foo'
    y = 42
    alpha = 123
    beta = 'asdf'
    beta_other = 999
    gamma = 'qwerty'
    alpha_yetanother = 'meow'
    delta = 'moo'
    state = model.initial_state(
        batch_size,
        x,
        y,
        alpha=alpha,
        beta=beta,
        tag_kwargs=dict(
            other=dict(
                beta=beta_other,
                gamma=gamma
            ),
            yetanother=dict(
                alpha=alpha_yetanother,
                delta=delta
            )
        )
    )
    output = model(
        input_sequence,
        x,
        y,
        alpha=alpha,
        beta=beta,
        include_first=False,
        tag_kwargs=dict(
            other=dict(
                beta=beta_other,
                gamma=gamma
            ),
            yetanother=dict(
                alpha=alpha_yetanother,
                delta=delta
            )
        )
    )
