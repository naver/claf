
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def weight(module):
    """
    weight initialization (according to module type)

    * Args:
        module: torch.nn.Module
    """

    if type(module) == list:
        for m in module:
            weight(m)

    if isinstance(module, nn.Conv2d):
        logger.info("initializing Conv Layer")
        torch.nn.init.uniform_(module.weight)

    elif isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        logger.info("Initializing Linear Layer")

    elif isinstance(module, nn.GRU):
        torch.nn.init.normal_(module.weight_hh_l0, std=0.05)
        logger.info("Initializing GRU Layer")
