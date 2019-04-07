
import logging
import os
import sys

import numpy as np
import torch

from claf.learn.mode import Mode


""" Interface """


def get_user_input(category):
    print(f"{category.capitalize()} > ", end="")
    sys.stdout.flush()

    user_input = sys.stdin.readline()
    try:
        return eval(user_input)
    except BaseException:
        return str(user_input)


def flatten(l):
    for item in l:
        if isinstance(item, list):
            for in_item in flatten(item):
                yield in_item
        else:
            yield item


def serializable(tensor):
    if isinstance(tensor, (torch.Tensor, torch.autograd.Variable)) and tensor.dim() > 1:
        return tensor.data.cpu().numpy().tolist(),
    elif isinstance(tensor, (torch.Tensor, torch.autograd.Variable)):
        return tensor.item()
    elif isinstance(tensor, np.ndarray):
        return tensor.tolist()
    elif isinstance(tensor, (list, tuple)):
        return [serializable(item) for item in tensor]
    elif isinstance(tensor, dict):
        return {key: serializable(value) for key, value in tensor.items()}
    else:
        return tensor


""" Logging """


def set_logging_config(mode, config):
    stdout_handler = logging.StreamHandler(sys.stdout)

    logging_handlers = [stdout_handler]
    logging_level = logging.INFO

    if mode == Mode.TRAIN:
        log_path = os.path.join(
            config.trainer.log_dir, f"{config.data_reader.dataset}_{config.model.name}.log"
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        logging_handlers.append(file_handler)
    elif mode == Mode.PREDICT:
        logging_level = logging.WARNING

    logging.basicConfig(
        format="%(asctime)s (%(filename)s:%(lineno)d): [%(levelname)s] - %(message)s",
        handlers=logging_handlers,
        level=logging_level,
    )
