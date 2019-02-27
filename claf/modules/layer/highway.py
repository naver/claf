
import torch
import torch.nn as nn

from claf.modules.activation import get_activation_fn


class Highway(nn.Module):
    """
    Highway Networks (https://arxiv.org/abs/1505.00387)
    https://github.com/allenai/allennlp/blob/master/allennlp/modules/highway.py

    * Args:
        input_size: The number of expected features in the input `x`
        num_layers: The number of Highway layers.
        activation: Activation Function (ReLU is default)
    """

    def __init__(self, input_size, num_layers=2, activation="relu"):
        super(Highway, self).__init__()
        self.activation_fn = activation
        if type(activation) == str:
            self.activation_fn = get_activation_fn(activation)()
        self._layers = torch.nn.ModuleList(
            [nn.Linear(input_size, input_size * 2) for _ in range(num_layers)]
        )

        for layer in self._layers:
            layer.bias[input_size:].data.fill_(
                1
            )  # should bias the highway layer to just carry its input forward.

    def forward(self, x):
        current_input = x
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input

            nonlinear_part, gate = projected_input.chunk(2, dim=-1)
            nonlinear_part = self.activation_fn(nonlinear_part)
            gate = torch.sigmoid(gate)

            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input
