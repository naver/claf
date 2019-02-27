import torch
import torch.nn as nn

from claf.modules.layer.normalization import LayerNorm


class ResidualConnection(nn.Module):
    """
    ResidualConnection
        in Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)

    => f(x) + x

    * Args:
        dim: the number of dimension

    * Kwargs:
        layer_dropout: layer dropout probability (stochastic depth)
        dropout: dropout probability
    """

    def __init__(self, dim, layer_dropout=None, layernorm=False):
        super(ResidualConnection, self).__init__()

        self.survival = None
        if layer_dropout < 1:
            self.survival = torch.FloatTensor([layer_dropout])
        if layernorm:
            self.norm = LayerNorm(dim)
        else:
            self.norm = lambda x: x

    def forward(self, x, sub_layer_fn):
        # implementation of stochastic depth
        if self.training and self.survival is not None:
            survival_prob = torch.bernoulli(self.survival).item()
            if survival_prob == 1:
                return x + sub_layer_fn(self.norm(x))
            else:
                return x
        else:
            return x + sub_layer_fn(self.norm(x))
