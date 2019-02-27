
from overrides import overrides

import torch
from torch.autograd import Variable

from claf.data import utils
from claf.data.batch import make_batch


class PadCollator:
    """
    Collator apply pad and make tensor
    Minimizes amount of padding needed while producing mini-batch.

    * Kwargs:
        cuda_device_id: tensor assign to cuda device id
            Default is None (CPU)
        skip_keys: skip to make tensor
    """

    def __init__(self, cuda_device_id=None, skip_keys=["text"]):
        self.cuda_device_id = cuda_device_id
        self.skip_keys = skip_keys

    def __call__(self, features, labels):
        self.collate(features)
        self.collate(labels, apply_pad=False)

        return make_batch(features, labels)

    def collate(self, datas, apply_pad=True, pad_value=0):
        for data_name, data in datas.items():
            if isinstance(data, dict):
                for key, value in data.items():
                    data[key] = self._collate(
                        value, apply_pad=apply_pad, token_name=key, pad_value=pad_value)
            else:
                datas[data_name] = self._collate(data, apply_pad=apply_pad)

    def _collate(self, value, apply_pad=True, token_name=None, pad_value=0):
        if apply_pad:
            value = self._apply_pad(value, token_name=token_name, pad_value=pad_value)
        return self._make_tensor(value)

    def _apply_pad(self, value, token_name=None, pad_value=0):
        return utils.padding_tokens(value, token_name=token_name, pad_value=pad_value)

    def _make_tensor(self, value):
        if not isinstance(value, torch.Tensor):
            value_type = utils.get_token_type(value)
            if value_type == int:
                value = torch.LongTensor(value)
            else:
                value = torch.FloatTensor(value)

        value = Variable(value, requires_grad=False)
        if self.cuda_device_id is not None:
            value = value.cuda(self.cuda_device_id)
        return value


class FeatLabelPadCollator(PadCollator):
    """
    Collator apply pad and make tensor
    Minimizes amount of padding needed while producing mini-batch.

    FeatLabelPadCollator allows applying pad to not only features, but also labels.

    * Kwargs:
        cuda_device_id: tensor assign to cuda device id
            Default is None (CPU)
        skip_keys: skip to make tensor
    """

    @overrides
    def __call__(self, features, labels, apply_pad_labels=(), apply_pad_values=()):
        self.collate(features)
        self.collate(labels, apply_pad=False,
                     apply_pad_labels=apply_pad_labels, apply_pad_values=apply_pad_values)

        return make_batch(features, labels)

    @overrides
    def collate(self, datas, apply_pad=True, apply_pad_labels=(), apply_pad_values=()):
        for data_name, data in datas.items():
            if not apply_pad and data_name in apply_pad_labels:
                _apply_pad = True  # ignore apply_pad
                pad_value = apply_pad_values[apply_pad_labels.index(data_name)]
            else:
                _apply_pad = apply_pad
                pad_value = 0

            if isinstance(data, dict):
                for key, value in data.items():
                    data[key] = self._collate(
                        value, apply_pad=_apply_pad, token_name=key, pad_value=pad_value)
            else:
                datas[data_name] = self._collate(data, apply_pad=_apply_pad, pad_value=pad_value)
