
import torch.nn as nn
import torch.nn.functional as F

from .pointwise_conv import PointwiseConv


class DepSepConv(nn.Module):
    """
    Depthwise Separable Convolutions
        in Xception: Deep Learning with Depthwise Separable Convolutions (https://arxiv.org/abs/1610.02357)

    depthwise -> pointwise (1x1 conv)

    * Args:
        input_size: the number of input tensor's dimension
        num_filters: the number of convolution filter
        kernel_size: the number of convolution kernel size
    """

    def __init__(self, input_size=None, num_filters=None, kernel_size=None):
        super(DepSepConv, self).__init__()

        self.depthwise = nn.Conv1d(
            in_channels=input_size,
            out_channels=input_size,
            kernel_size=kernel_size,
            groups=input_size,
            padding=kernel_size // 2,
        )
        nn.init.kaiming_normal_(self.depthwise.weight)
        self.pointwise = PointwiseConv(input_size=input_size, num_filters=num_filters)
        self.activation_fn = F.relu

    def forward(self, x):
        x = self.depthwise(x.transpose(1, 2))
        x = self.pointwise(x.transpose(1, 2))
        x = self.activation_fn(x)
        return x
