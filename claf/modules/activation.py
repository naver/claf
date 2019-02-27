
import torch.nn as nn


def get_activation_fn(name):
    """ PyTorch built-in activation functions """

    activation_functions = {
        "linear": lambda: lambda x: x,
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "elu": nn.ELU,
        "prelu": nn.PReLU,
        "leaky_relu": nn.LeakyReLU,
        "threshold": nn.Threshold,
        "hardtanh": nn.Hardtanh,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "log_sigmoid": nn.LogSigmoid,
        "softplus": nn.Softplus,
        "softshrink": nn.Softshrink,
        "softsign": nn.Softsign,
        "tanhshrink": nn.Tanhshrink,
    }

    if name not in activation_functions:
        raise ValueError(
            f"'{name}' is not included in activation_functions. use below one. \n {activation_functions.keys()}"
        )

    return activation_functions[name]
