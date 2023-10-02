import torch

from torch_extras.layer import Layer
from torch_unidirectional import SimpleLayerUnidirectional
from .sublayer import get_unidirectional_sublayer

def get_feedforward_sublayer(d_model, feedforward_size, dropout):
    return get_unidirectional_sublayer(
        SimpleLayerUnidirectional(get_feedforward_module(
            d_model,
            feedforward_size,
            dropout
        )),
        d_model,
        dropout
    )

def get_feedforward_module(d_model, feedforward_size, dropout):
    dropout_module = [torch.nn.Dropout(dropout)] if dropout is not None else []
    return torch.nn.Sequential(
        Layer(d_model, feedforward_size, activation=torch.nn.ReLU(), bias=True),
        *dropout_module,
        Layer(feedforward_size, d_model, bias=True)
    )
