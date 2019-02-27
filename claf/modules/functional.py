"""
    some functional codes from allennlp: https://github.com/allenai/allennlp

    - add_masked_value : replace_masked_values (allennlp)
    - get_mask_from_tokens : get_mask_from_tokens (allennlp)
    - last_dim_masked_softmax : last_dim_masked_softmax (allennlp)
    - masked_softmax : masked_softmax (allennlp)
    - weighted_sum : weighted_sum (allennlp)
"""

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def add_masked_value(tensor, mask, value=-1e7):
    mask = mask.float()
    one_minus_mask = 1.0 - mask
    values_to_add = value * one_minus_mask
    return tensor * mask + values_to_add


def get_mask_from_tokens(tokens):
    tensor_dims = [(tensor.dim(), tensor) for tensor in tokens.values()]
    tensor_dims.sort(key=lambda x: x[0])

    smallest_dim = tensor_dims[0][0]
    if smallest_dim == 2:
        token_tensor = tensor_dims[0][1]
        return (token_tensor != 0).long()
    elif smallest_dim == 3:
        character_tensor = tensor_dims[0][1]
        return ((character_tensor > 0).long().sum(dim=-1) > 0).long()
    else:
        raise ValueError("Expected a tensor with dimension 2 or 3, found {}".format(smallest_dim))


def last_dim_masked_softmax(x, mask):
    x_shape = x.size()
    reshaped_x = x.view(-1, x.size()[-1])

    while mask.dim() < x.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(x).contiguous().float()
    mask = mask.view(-1, mask.size()[-1])

    reshaped_result = masked_softmax(reshaped_x, mask)
    return reshaped_result.view(*x_shape)


def masked_softmax(x, mask):
    if mask is None:
        raise ValueError("mask can't be None.")

    output = F.softmax(x * mask, dim=-1)
    output = output * mask
    output = output / (output.sum(dim=1, keepdim=True) + 1e-13)
    return output


def weighted_sum(attention, matrix):  # pragma: no cover
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    elif attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix)
    else:
        raise ValueError(
            f"attention dim {attention.dim()} and matrix dim {matrix.dim()} operation not support. (2, 3) and (3, 3) are available dimemsion."
        )


def masked_zero(tensor, mask):
    """ Tensor masking operation """
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(-1)

    if isinstance(tensor, torch.FloatTensor):
        mask = mask.float()
    elif isinstance(tensor, torch.ByteTensor):
        mask = mask.byte()
    elif isinstance(tensor, torch.LongTensor):
        mask = mask.long()

    return tensor * mask


def masked_log_softmax(vector, mask):  # pragma: no cover
    if mask is not None:
        vector = vector + mask.float().log()
    return torch.nn.functional.log_softmax(vector, dim=1)


def get_sorted_seq_config(features, pad_index=0):
    tensor_dims = [(tensor.dim(), tensor) for tensor in features.values()]
    tensor_dims.sort(key=lambda x: x[0])

    smallest_dim = tensor_dims[0][0]
    if smallest_dim == 2:
        token_tensor = tensor_dims[0][1]
    else:
        raise ValueError("features smallest_dim must be `2` ([B, S_L]) ")

    seq_lengths = torch.sum(token_tensor > pad_index, dim=-1)
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    _, unperm_idx = perm_idx.sort(0)

    return {"seq_lengths": seq_lengths, "perm_idx": perm_idx, "unperm_idx": unperm_idx}


def forward_rnn_with_pack(rnn_module, tensor, seq_config):
    sorted_tensor = tensor[seq_config["perm_idx"]]
    packed_input = pack_padded_sequence(sorted_tensor, seq_config["seq_lengths"], batch_first=True)
    packed_output, _ = rnn_module(packed_input)
    output, _ = pad_packed_sequence(packed_output, batch_first=True)
    output = output[seq_config["unperm_idx"]]  # restore origin order
    return output
