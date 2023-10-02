import math
import typing

import torch

class Layer(torch.nn.Module):
    """A fully-connected layer consisting of connection weights and an
    activation function.
    
    Treating these things as a unit is useful because the activation function
    can be used to initialize the weights with Xavier initialization
    properly."""

    def __init__(self, input_size: int, output_size: int,
            activation: torch.nn.Module=torch.nn.Identity(), bias: bool=True):
        """
        :param input_size: The number of units in the input to the layer.
        :param output_size: The number of units in the output of the layer.
        :param activation: The activation function. By default, no activation
            function is applied.
        :param bias: Whether to use a bias term.
        """
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size, bias=bias)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Let :math:`B` be batch size, :math:`X` be ``input_size``, and
        :math:`Y` be ``output_size``.

        :param x: A tensor of size :math:`B \times X`.
        :return: A tensor of size :math:`B \times Y`.
        """
        return self.activation(self.linear(x))

    def get_nonlinearity_name(self) -> str:
        """Get the name of the activation function as a string which can be
        used with :py:func:`~torch.nn.init.calculate_gain`."""
        a = self.activation
        if isinstance(a, (torch.nn.Identity, torch.nn.Softmax)):
            return 'linear'
        else:
            return type(a).__name__.lower()

    def get_gain(self) -> float:
        """Get the correct gain value for initialization based on the
        activation function."""
        return torch.nn.init.calculate_gain(self.get_nonlinearity_name())

    def xavier_uniform_init(self,
            generator: typing.Optional[torch.Generator]=None) -> None:
        """Initialize the parameters of the layer using Xavier initialization.
        The correct gain is used based on the activation function. The bias
        term, if it exists, will be initialized to 0.
        
        :param generator: Optional PyTorch RNG.
        """
        xavier_uniform_(
            self.linear.weight,
            self.fan_in_size(),
            self.fan_out_size(),
            self.get_gain(),
            generator=generator
        )
        if self.linear.bias is not None:
            torch.nn.init.constant_(self.linear.bias, 0.0)

    def fan_in_size(self) -> int:
        return self.input_size()

    def fan_out_size(self) -> int:
        return self.output_size()

    def input_size(self) -> int:
        """Get the size of the input to this layer."""
        return self.linear.in_features

    def output_size(self) -> int:
        """Get the size of the output of this layer."""
        return self.linear.out_features

class FeedForward(torch.nn.Sequential):
    """Multiple :py:class:`Layer`s in serial, forming a feed-forward neural
    network."""

    def __init__(self, input_size: int, layer_sizes: typing.Iterable[int],
            activation: torch.nn.Module, bias: bool=True):
        """
        :param input_size: The number of units in the input to the first layer.
        :param layer_sizes: The sizes of the outputs of each layer, including
            the last.
        :param activation: The activation function applied to the output of
            each layer. This should be a non-linear function, since multiple
            linear transformations is equivalent to a single linear
            transformation anyway.
        :param bias: Whether to use a bias term in each layer.
        """
        modules = []
        for layer_size in layer_sizes:
            modules.append(Layer(input_size, layer_size, activation=activation, bias=bias))
            input_size = layer_size
        super().__init__(*modules)

    def input_size(self) -> int:
        """The size of the input to this network."""
        return self[0].input_size()

    def output_size(self) -> int:
        """The size of the output of this network."""
        return self[-1].output_size()

class MultiLayer(Layer):
    """A module representing :math:`num_layers` fully-connected layers all with
    the same input and activation function. The layer outputs will be computed
    in parallel."""

    def __init__(self, input_size: int, output_size: int, num_layers: int,
            activation: torch.nn.Module=torch.nn.Identity(), bias: bool=True):
        """
        :param input_size: The number of units in the input to the layers.
        :param output_size: The number of units in the output of each layer.
        :param n: The number of layers.
        :param activation: The activation function. By default, no activation
            function is applied.
        :param bias: Whether to use a bias term.
        """
        super().__init__(
            input_size,
            num_layers * output_size,
            activation=activation,
            bias=bias
        )
        self._output_size = output_size
        self._num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Let :math:`B` be batch size, :math:`X` be ``input_size``,
        :math:`Y` be ``output_size``, and :math:`n` be the number of layers.

        :param x: A tensor of size :math:`B \times X`.
        :return: A tensor of size :math:`B \times n \times Y`.
        """
        # y : B x (n * Y)
        y = self.linear(x)
        # y_view : B x n x Y
        y_view = y.view(x.size(0), self._num_layers, self._output_size)
        # return : B x n x Y
        return self.activation(y_view)

    def fan_out_size(self) -> int:
        return self._output_size

    def output_size(self) -> typing.Tuple[int, int]:
        return (self._num_layers, self._output_size)

def xavier_uniform_(
        tensor: torch.Tensor,
        fan_in: int,
        fan_out: int,
        gain: float,
        generator: typing.Optional[torch.Generator]
    ):
    """A rewrite of :py:func:`~torch.nn.init.xavier_uniform` that accepts a
    RNG and works on multi-dimensional tensors."""
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-a, a, generator=generator)
