
from pytorch_pretrained_bert.optimization import BertAdam
import torch


def get_optimizer_by_name(name):
    optimizers = {
        "adam": torch.optim.Adam,
        "bert_adam": BertAdam,
        "sparse_adam": torch.optim.SparseAdam,
        "adagrad": torch.optim.Adagrad,
        "adadelta": torch.optim.Adadelta,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "adamax": torch.optim.Adamax,
        "averaged_sgd": torch.optim.ASGD,
    }

    if name in optimizers:
        return optimizers[name]
    else:
        raise ValueError(f"'{name}' is not registered. \noptimizer list: {list(optimizers.keys())}")
