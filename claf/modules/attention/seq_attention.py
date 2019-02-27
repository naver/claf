#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
original code from: https://github.com/facebookresearch/DrQA/blob/master/drqa/reader/layers.py
"""

import torch.nn as nn
import torch.nn.functional as F


class SeqAttnMatch(nn.Module):
    """
    Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, embed_dim, identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(embed_dim, embed_dim)
        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        scores = x_proj.bmm(y_proj.transpose(2, 1))

        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores = scores.masked_fill((y_mask == 0), -1e30)

        alpha_flat = F.softmax(scores.view(-1, y.size(1)), -1)
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))

        matched_seq = alpha.bmm(y)
        return matched_seq


class LinearSeqAttn(nn.Module):
    """
    Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_((x_mask == 0), -1e30)
        alpha = F.softmax(scores, dim=-1)
        return alpha


class BilinearSeqAttn(nn.Module):
    """
    A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.
    Optionally don't normalize output weights.
    """

    def __init__(self, x_size, y_size, identity=False, normalize=True):
        super(BilinearSeqAttn, self).__init__()
        self.normalize = normalize

        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_((x_mask == 0), -1e30)
        if self.normalize:
            if self.training:
                alpha = F.log_softmax(xWy, dim=-1)
            else:
                alpha = F.softmax(xWy, dim=-1)
        else:
            alpha = xWy.exp()
        return alpha
