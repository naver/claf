
import torch.nn as nn
import torch.nn.functional as F

from claf.modules.conv import PointwiseConv


class PositionwiseFeedForward(nn.Module):
    """
    Pointwise Feed-Forward Layer

    * Args:
        input_size: the number of input size
        hidden_size: the number of hidden size

    * Kwargs:
        dropout: the probability of dropout
    """

    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.pointwise_conv1 = PointwiseConv(input_size=input_size, num_filters=hidden_size)
        self.pointwise_conv2 = PointwiseConv(input_size=hidden_size, num_filters=input_size)
        self.activation_fn = F.relu
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.pointwise_conv1(x)
        x = self.activation_fn(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x
