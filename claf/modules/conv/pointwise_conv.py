
import torch
import torch.nn as nn


class PointwiseConv(nn.Module):
    """
    Pointwise Convolution (1x1 Conv)

    Convolution 1 Dimension (Faster version)
    (cf. https://github.com/huggingface/pytorch-openai-transformer-lm/blob/\
        eafc28abdfadfa0732f03a0fc65805c5bfb2ffe7/model_pytorch.py#L45)

    * Args:
        input_size: the number of input tensor's dimension
        num_filters: the number of convolution filter
    """

    # nf: num_filters, rf: kernel_size, nx: in_channels
    def __init__(self, input_size, num_filters):
        super(PointwiseConv, self).__init__()

        self.kernel_size = 1
        self.num_filters = num_filters

        weight = torch.empty(input_size, num_filters)
        nn.init.normal_(weight, std=0.02)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(num_filters))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.num_filters,)
        x = torch.addmm(self.bias, x.contiguous().view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x
