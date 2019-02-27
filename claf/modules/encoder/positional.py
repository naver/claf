
import math
import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional Encoding
        in "Attention is All You Need" (https://arxiv.org/abs/1706.03762)

    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    expressed in terms of y, sin(x) and cos(x).

    (cf. https://github.com/tensorflow/tensor2tensor/blob/42c3f377f441e5a0f431127d63e71414ead291c4/\
        tensor2tensor/layers/common_attention.py#L388)

    * Args:
        embed_dim: the number of embedding dimension

    * Kwargs:
        max_len: the number of maximum sequence length
    """

    def __init__(self, embed_dim, max_length=2000):
        super(PositionalEncoding, self).__init__()
        signal_sinusoid = self._get_timing_signal(max_length, embed_dim)

        self.register_buffer("position_encoding", signal_sinusoid)

    def _get_timing_signal(self, length, channels, min_timescale=1.0, max_timescale=1.0e4):
        position = np.arange(length)
        num_timescales = channels // 2
        log_timescale_increment = math.log(
            float(max_timescale) / float(min_timescale) / (float(num_timescales) - 1)
        )
        inv_timescales = min_timescale * np.exp(
            np.arange(num_timescales).astype(np.float) * -log_timescale_increment
        )
        scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

        signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
        signal = np.pad(signal, [[0, 0], [0, channels % 2]], "constant", constant_values=[0.0, 0.0])
        signal = signal.reshape([1, length, channels])

        return torch.from_numpy(signal).type(torch.FloatTensor)

    def forward(self, x):
        x = x + self.position_encoding[:, : x.size(1)]
        return x
