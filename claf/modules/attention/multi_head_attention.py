
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import claf.modules.functional as f


class MultiHeadAttention(nn.Module):
    """
    Transformer's Multi-Head Attention
        in "Attention is All You Need" (https://arxiv.org/abs/1706.03762)

    * Kwargs:
        num_head: the number of Head
        model_dim: the number of model dimension
        linear_key_dim: the number of linear key dimemsion
        linear_value_dim: the number of linear value dimension
    """

    def __init__(
        self, num_head=8, model_dim=100, dropout=0.1, linear_key_dim=None, linear_value_dim=None
    ):
        super(MultiHeadAttention, self).__init__()
        if linear_key_dim is None:
            linear_key_dim = model_dim
        if linear_value_dim is None:
            linear_value_dim = model_dim

        assert linear_key_dim % num_head == 0
        assert linear_value_dim % num_head == 0

        self.model_dim = model_dim
        self.num_head = num_head
        self.projection = nn.ModuleList(
            [
                nn.Linear(model_dim, linear_key_dim, bias=False),  # query
                nn.Linear(model_dim, linear_key_dim, bias=False),  # key
                nn.Linear(model_dim, linear_value_dim, bias=False),  # value
            ]
        )
        self.out_linear = nn.Linear(linear_value_dim, model_dim)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = lambda x: x

    def forward(self, q, k, v, mask=None):
        q, k, v = self._linear_projection(q, k, v)
        qs, ks, vs = self._split_heads(q, k, v)
        outputs = self._scaled_dot_product(qs, ks, vs, mask=mask)
        output = self._concat_heads(outputs)
        return self.out_linear(output)

    def _linear_projection(self, query, key, value):
        q = self.projection[0](query)
        k = self.projection[1](key)
        v = self.projection[2](value)
        return q, k, v

    def _split_heads(self, query, key, value):
        B = query.size(0)
        qs, ks, vs = [
            x.view(B, -1, self.num_head, x.size(-1) // self.num_head).transpose(1, 2)
            for x in [query, key, value]
        ]
        return qs, ks, vs

    def _scaled_dot_product(self, query, key, value, mask=None):
        K_D = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(K_D)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [B, #H, C_L, D]
            scores = f.add_masked_value(scores, mask, value=-1e7)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, value)

    def _concat_heads(self, outputs):
        B = outputs.size(0)
        num_head, dim = outputs.size()[-2:]

        return outputs.transpose(1, 2).contiguous().view(B, -1, self.num_head * dim)
