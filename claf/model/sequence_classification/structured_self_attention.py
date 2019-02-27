
from overrides import overrides

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from claf.decorator import register
from claf.model.base import ModelWithTokenEmbedder
from claf.model.sequence_classification.mixin import SequenceClassification
from claf.modules import functional as f


@register("model:structured_self_attention")
class StructuredSelfAttention(SequenceClassification, ModelWithTokenEmbedder):
    """
    Implementation of model presented in
    A Structured Self-attentive Sentence Embedding
    (https://arxiv.org/abs/1703.03130)

    * Args:
        token_embedder: used to embed the sequence
        num_classes: number of classified classes

    * Kwargs:
        encoding_rnn_hidden_dim: hidden dimension of rnn (unidirectional)
        encoding_rnn_num_layer: the number of rnn layers
        encoding_rnn_dropout: rnn dropout probability
        attention_dim: attention dimension  # d_a in the paper
        num_attention_heads: number of attention heads  # r in the paper
        sequence_embed_dim: dimension of sequence embedding
        dropout: classification layer dropout
        penalization_coefficient: penalty coefficient for frobenius norm
    """

    def __init__(
        self,
        token_embedder,
        num_classes,
        encoding_rnn_hidden_dim=300,
        encoding_rnn_num_layer=2,
        encoding_rnn_dropout=0.,
        attention_dim=350,
        num_attention_heads=30,
        sequence_embed_dim=2000,
        dropout=0.5,
        penalization_coefficient=1.,
    ):
        super(StructuredSelfAttention, self).__init__(token_embedder)

        rnn_input_dim = token_embedder.get_embed_dim()

        self.num_classes = num_classes

        self.encoding_rnn_hidden_dim = encoding_rnn_hidden_dim * 2  # bidirectional
        self.attention_dim = attention_dim
        self.num_attention_heads = num_attention_heads
        self.project_dim = sequence_embed_dim
        self.dropout = dropout
        self.penalization_coefficient = penalization_coefficient

        self.encoder = nn.LSTM(
            input_size=rnn_input_dim,
            hidden_size=encoding_rnn_hidden_dim,
            num_layers=encoding_rnn_num_layer,
            dropout=encoding_rnn_dropout,
            bidirectional=True,
            batch_first=True,
        )

        self.A = nn.Sequential(
            nn.Linear(self.encoding_rnn_hidden_dim, attention_dim, bias=False),
            nn.Tanh(),
            nn.Linear(attention_dim, num_attention_heads, bias=False),
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(self.encoding_rnn_hidden_dim * num_attention_heads, sequence_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(sequence_embed_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    @overrides
    def forward(self, features, labels=None):
        """
        * Args:
            features: feature dictionary like below.
            {"sequence": [0, 3, 4, 1]}

        * Kwargs:
            label: label dictionary like below.
            {"class_idx": 2, "data_idx": 0}
             Do not calculate loss when there is no label. (inference/predict mode)

        * Returns: output_dict (dict) consisting of
            - sequence_embed: embedding vector of the sequence
            - class_logits: representing unnormalized log probabilities of the class.

            - class_idx: target class idx
            - data_idx: data idx
            - loss: a scalar loss to be optimized
        """

        sequence = features["sequence"]

        # Sorted Sequence config (seq_lengths, perm_idx, unperm_idx) for RNN pack_forward
        sequence_config = f.get_sorted_seq_config(sequence)

        token_embed = self.token_embedder(sequence)

        token_encodings = f.forward_rnn_with_pack(
            self.encoder, token_embed, sequence_config
        )  # [B, L, encoding_rnn_hidden_dim]

        attention = self.A(token_encodings).transpose(1, 2)  # [B, num_attention_heads, L]

        sequence_mask = f.get_mask_from_tokens(sequence).float()  # [B, L]
        sequence_mask = sequence_mask.unsqueeze(1).expand_as(attention)
        attention = F.softmax(f.add_masked_value(attention, sequence_mask) + 1e-13, dim=2)

        attended_encodings = torch.bmm(
            attention, token_encodings
        )  # [B, num_attention_heads, sequence_embed_dim]
        sequence_embed = self.fully_connected(
            attended_encodings.view(attended_encodings.size(0), -1)
        )  # [B, sequence_embed_dim]

        class_logits = self.classifier(sequence_embed)  # [B, num_classes]

        output_dict = {"sequence_embed": sequence_embed, "class_logits": class_logits}

        if labels:
            class_idx = labels["class_idx"]
            data_idx = labels["data_idx"]

            output_dict["class_idx"] = class_idx
            output_dict["data_idx"] = data_idx

            # Loss
            loss = self.criterion(class_logits, class_idx)
            loss += self.penalty(attention)
            output_dict["loss"] = loss.unsqueeze(0)  # NOTE: DataParallel concat Error

        return output_dict

    def penalty(self, attention):
        aa = torch.bmm(
            attention, attention.transpose(1, 2)
        )  # [B, num_attention_heads, num_attention_heads]
        penalization_term = ((aa - aa.new_tensor(np.eye(aa.size(1)))) ** 2).sum() ** 0.5
        return penalization_term * self.penalization_coefficient
