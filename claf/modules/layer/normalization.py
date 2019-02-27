
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer Normalization
    (https://arxiv.org/abs/1607.06450)
    """

    def __init__(self, normalized_shape, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
