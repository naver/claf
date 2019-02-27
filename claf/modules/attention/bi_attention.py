
import torch
import torch.nn as nn

import claf.modules.functional as f


class BiAttention(nn.Module):
    """
    Attention Flow Layer
        in BiDAF (https://arxiv.org/pdf/1611.01603.pdf)

    The Similarity matrix
    Context-to-query Attention (C2Q)
    Query-to-context Attention (Q2C)

    * Args:
        model_dim: The number of module dimension
    """

    def __init__(self, model_dim):
        super(BiAttention, self).__init__()
        self.model_dim = model_dim
        self.W = nn.Linear(6 * model_dim, 1, bias=False)

    def forward(self, context, context_mask, query, query_mask):
        c, c_mask, q, q_mask = context, context_mask, query, query_mask

        S = self._make_similiarity_matrix(c, q)  # (B, C_L, Q_L)
        masked_S = f.add_masked_value(S, query_mask.unsqueeze(1), value=-1e7)

        c2q = self._context2query(S, q, q_mask)
        q2c = self._query2context(masked_S.max(dim=-1)[0], c, c_mask)

        # [h; u˜; h◦u˜; h◦h˜] ~ (B, C_L, 8d)
        G = torch.cat((c, c2q, c * c2q, c * q2c), dim=-1)
        return G

    def _make_similiarity_matrix(self, c, q):
        # B: batch_size, C_L: context_maxlen, Q_L: query_maxlen
        B, C_L, Q_L = c.size(0), c.size(1), q.size(1)

        matrix_shape = (B, C_L, Q_L, self.model_dim * 2)

        c_aug = c.unsqueeze(2).expand(matrix_shape)  # (B, C_L, Q_L, 2d)
        q_aug = q.unsqueeze(1).expand(matrix_shape)  # (B, C_L, Q_L, 2d)

        c_q = torch.mul(c_aug, q_aug)  # element-wise multiplication

        concated_vector = torch.cat((c_aug, q_aug, c_q), dim=3)  # [h; u; h◦u]
        return self.W(concated_vector).view(c.size(0), C_L, Q_L)

    def _context2query(self, S, q, q_mask):
        attention = f.last_dim_masked_softmax(S, q_mask)  # (B, C_L, Q_L)
        c2q = f.weighted_sum(attention=attention, matrix=q)  # (B, C_L, 2d)

        return c2q

    def _query2context(self, S, c, c_mask):
        attention = f.masked_softmax(S, c_mask)  # (B, C_L)
        q2c = f.weighted_sum(attention=attention, matrix=c)

        return q2c.unsqueeze(1).expand(c.size())  # (B, C_L, 2d)
