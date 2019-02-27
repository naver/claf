
from argparse import Namespace
import copy
import json

import jsbeautifier
import numpy as np
import random
import torch


def pretty_json_dumps(inputs):
    js_opts = jsbeautifier.default_options()
    js_opts.indent_size = 2

    inputs = remove_none(inputs)
    return jsbeautifier.beautify(json.dumps(inputs))


def remove_none(obj):
    if isinstance(obj, (list, tuple, set)):
        return type(obj)(remove_none(x) for x in obj if x is not None)
    elif isinstance(obj, dict):
        return type(obj)(
            (remove_none(k), remove_none(v))
            for k, v in obj.items()
            if k is not None and v is not None
        )
    else:
        return obj


def convert_config2dict(config):
    config_dict = copy.deepcopy(config)
    if isinstance(config_dict, Namespace):
        config_dict = vars(config_dict)

    for k, v in config_dict.items():
        if isinstance(v, Namespace):
            config_dict[k] = convert_config2dict(v)
    return config_dict


def set_global_seed(seed=21):
    # Tensorflow
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # NumPy
    np.random.seed(seed)

    # Python
    random.seed(seed)
