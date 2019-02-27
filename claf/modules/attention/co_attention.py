
import torch
import torch.nn as nn
import torch.nn.functional as F

import claf.modules.functional as f


class CoAttention(nn.Module):
    """
    CoAttention encoder
        in Dynamic Coattention Networks For Question Answering (https://arxiv.org/abs/1611.01604)

    check the Figure 2 in paper

    * Args:
        embed_dim: the number of input embedding dimension
    """

    def __init__(self, embed_dim):
        super(CoAttention, self).__init__()

        self.W_0 = nn.Linear(embed_dim * 3, 1, bias=False)

    def forward(self, context_embed, question_embed, context_mask=None, question_mask=None):
        C, Q = context_embed, question_embed
        B, C_L, Q_L, D = C.size(0), C.size(1), Q.size(1), Q.size(2)

        similarity_matrix_shape = torch.zeros(B, C_L, Q_L, D)  # (B, C_L, Q_L, D)

        C_ = C.unsqueeze(2).expand_as(similarity_matrix_shape)
        Q_ = Q.unsqueeze(1).expand_as(similarity_matrix_shape)
        C_Q = torch.mul(C_, Q_)

        S = self.W_0(torch.cat([C_, Q_, C_Q], 3)).squeeze(3)  # (B, C_L, Q_L)

        S_question = S
        if question_mask is not None:
            S_question = f.add_masked_value(S_question, question_mask.unsqueeze(1), value=-1e7)
        S_q = F.softmax(S_question, 2)  # (B, C_L, Q_L)

        S_context = S.transpose(1, 2)
        if context_mask is not None:
            S_context = f.add_masked_value(S_context, context_mask.unsqueeze(1), value=-1e7)
        S_c = F.softmax(S_context, 2)  # (B, Q_L, C_L)

        A = torch.bmm(S_q, Q)  # context2query (B, C_L, D)
        B = torch.bmm(S_q, S_c).bmm(C)  # query2context (B, Q_L, D)
        out = torch.cat([C, A, C * A, C * B], dim=-1)
        return out
