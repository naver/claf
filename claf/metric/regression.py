
import numpy as np


def mse(outputs, labels):
    if type(outputs) == list:
        outputs = np.array(outputs)
    if type(labels) == list:
        labels = np.array(labels)

    # read prediction and compute result
    if outputs.ndim != 1:
        outputs = outputs.reshape(-1)
    if labels.ndim != 1:
        labels = labels.reshape(-1)

    return np.square(labels.astype(np.float32) - outputs).sum()
