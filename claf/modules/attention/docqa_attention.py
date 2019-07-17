import torch
import torch.nn as nn
import torch.nn.functional as F

from claf.modules import initializer
import claf.modules.functional as f


class DocQAAttention(nn.Module):
    """
        Bi-Attention Layer + (Self-Attention)
            in DocumentQA (https://arxiv.org/abs/1710.10723)

        * Args:
            rnn_dim: the number of GRU cell hidden size
            linear_dim: the number of linear hidden size

        * Kwargs:
            self_attn: (bool) self-attention
            weight_init: (bool) weight initialization

    """

    def __init__(self, rnn_dim, linear_dim, self_attn=False, weight_init=True):
        super(DocQAAttention, self).__init__()
        self.self_attn = self_attn

        self.input_w = nn.Linear(2 * rnn_dim, 1, bias=False)
        self.key_w = nn.Linear(2 * rnn_dim, 1, bias=False)

        self.dot_w = nn.Parameter(torch.randn(1, 1, rnn_dim * 2))
        torch.nn.init.xavier_uniform_(self.dot_w)

        self.bias = nn.Parameter(torch.FloatTensor([[1]]))
        self.diag_mask = nn.Parameter(torch.eye(5000))  # NOTE: (hard-code) max_sequence_length

        if weight_init:
            initializer.weight(self.input_w)
            initializer.weight(self.key_w)

    def forward(self, x, x_mask, key, key_mask):
        S = self._trilinear(x, key)

        if self.self_attn:
            seq_length = x.size(1)
            diag_mask = self.diag_mask.narrow(0, 0, seq_length).narrow(1, 0, seq_length)
            joint_mask = 1 - self._compute_attention_mask(x_mask, key_mask)
            mask = torch.clamp(diag_mask + joint_mask, 0, 1)
            masked_S = S + mask * (-1e7)
            x2key = self._x2key(masked_S, key, key_mask)
            return torch.cat((x, x2key, x * x2key), dim=-1)
        else:
            joint_mask = 1 - self._compute_attention_mask(x_mask, key_mask)
            masked_S = S + joint_mask * (-1e7)
            x2key = self._x2key(masked_S, key, key_mask)

            masked_S = f.add_masked_value(S, key_mask.unsqueeze(1), value=-1e7)
            key2x = self._key2x(masked_S.max(dim=-1)[0], x, x_mask)
            return torch.cat((x, x2key, x * x2key, x * key2x), dim=-1)

    def _compute_attention_mask(self, x_mask, key_mask):
        x_mask = x_mask.unsqueeze(2)
        key_mask = key_mask.unsqueeze(1)
        joint_mask = torch.mul(x_mask, key_mask)
        return joint_mask

    def _trilinear(self, x, key):
        B, X_L, K_L = x.size(0), x.size(1), key.size(1)

        matrix_shape = (B, X_L, K_L)
        x_logits = self.input_w(x).expand(matrix_shape)
        key_logits = self.key_w(key).transpose(1, 2).expand(matrix_shape)

        x_dots = torch.mul(x, self.dot_w)
        x_key = torch.matmul(x_dots, key.transpose(1, 2))

        return x_logits + key_logits + x_key

    def _x2key(self, S, key, key_mask):
        if self.self_attn:
            bias = torch.exp(self.bias)
            S = torch.exp(S)
            attention = S / (S.sum(dim=-1, keepdim=True).expand(S.size()) + bias.expand(S.size()))
        else:
            attention = F.softmax(S, dim=-1)  # (B, C_L, Q_L)

        x2key = f.weighted_sum(attention=attention, matrix=key)  # (B, C_L, 2d)
        return x2key

    def _key2x(self, S, x, x_mask):
        attention = f.masked_softmax(S, x_mask)  # (B, C_L)
        key2x = f.weighted_sum(attention=attention, matrix=x)
        return key2x.unsqueeze(1).expand(x.size())  # (B, C_L, 2d)
