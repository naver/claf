import torch
import torch.nn as nn

from overrides import overrides

from claf.decorator import register
from claf.model.base import ModelWithTokenEmbedder
from claf.model.reading_comprehension.mixin import SQuADv1
import claf.modules.functional as f
import claf.modules.attention as attention
import claf.modules.encoder as encoder
import claf.modules.conv as conv
import claf.modules.layer as layer


@register("model:qanet")
class QANet(SQuADv1, ModelWithTokenEmbedder):
    """
        Document Reader Model. `Span Detector`

        Implementation of model presented in
        QANet:Combining Local Convolution with Global Self-Attention for Reading Comprehension
        (https://arxiv.org/abs/1804.09541)

        - Input Embedding Layer
        - Embedding Encoder Layer
        - Context-Query Attention Layer
        - Model Encoder Layer
        - Output Layer

        * Args:
            token_embedder: 'QATokenEmbedder', Used to embed the 'context' and 'question'.

        * Kwargs:
            lang_code: Dataset language code [en|ko]
            aligned_query_embedding: f_align(p_i) = sum(a_ij, E(qj), where the attention score a_ij
                captures the similarity between pi and each question words q_j.
                these features add soft alignments between similar but non-identical words (e.g., car and vehicle)
                it only apply to 'context_embed'.
            answer_maxlen: the most probable answer span of length less than or equal to {answer_maxlen}
            model_dim: the number of model dimension

            * Encoder Block Parameters (embedding, modeling)
              kernel_size: convolution kernel size in encoder block
              num_head: the number of multi-head attention's head
              num_conv_block: the number of convolution block in encoder block
                  [Layernorm -> Conv (residual)]
              num_encoder_block: the number of the encoder block
                  [position_encoding -> [n repeat conv block] -> Layernorm -> Self-attention (residual)
                   -> Layernorm -> Feedforward (residual)]

            dropout: the dropout probability
            layer_dropout: the layer dropout probability
                (cf. Deep Networks with Stochastic Depth(https://arxiv.org/abs/1603.09382) )
    """

    def __init__(
        self,
        token_embedder,
        lang_code="en",
        aligned_query_embedding=False,
        answer_maxlen=None,
        model_dim=128,
        kernel_size_in_embedding=7,
        num_head_in_embedding=8,
        num_conv_block_in_embedding=4,
        num_embedding_encoder_block=1,
        kernel_size_in_modeling=5,
        num_head_in_modeling=8,
        num_conv_block_in_modeling=2,
        num_modeling_encoder_block=7,
        dropout=0.1,
        layer_dropout=0.9,
    ):
        super(QANet, self).__init__(token_embedder)

        self.lang_code = lang_code
        self.aligned_query_embedding = aligned_query_embedding
        self.answer_maxlen = answer_maxlen
        self.token_embedder = token_embedder

        context_embed_dim, query_embed_dim = token_embedder.get_embed_dim()

        if self.aligned_query_embedding:
            context_embed_dim += query_embed_dim

        if context_embed_dim != query_embed_dim:
            self.context_highway = layer.Highway(context_embed_dim)
            self.query_highway = layer.Highway(query_embed_dim)

            self.context_embed_pointwise_conv = conv.PointwiseConv(context_embed_dim, model_dim)
            self.query_embed_pointwise_conv = conv.PointwiseConv(query_embed_dim, model_dim)
        else:
            highway = layer.Highway(context_embed_dim)

            self.context_highway = highway
            self.query_highway = highway

            embed_pointwise_conv = conv.PointwiseConv(context_embed_dim, model_dim)

            self.context_embed_pointwise_conv = embed_pointwise_conv
            self.query_embed_pointwise_conv = embed_pointwise_conv

        self.embed_encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    model_dim=model_dim,
                    kernel_size=kernel_size_in_embedding,
                    num_head=num_head_in_embedding,
                    num_conv_block=num_conv_block_in_modeling,
                    dropout=dropout,
                    layer_dropout=layer_dropout,
                )
                for _ in range(num_embedding_encoder_block)
            ]
        )

        self.co_attention = attention.CoAttention(model_dim)

        self.pointwise_conv = conv.PointwiseConv(model_dim * 4, model_dim)
        self.model_encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    model_dim=model_dim,
                    kernel_size=kernel_size_in_modeling,
                    num_head=num_head_in_modeling,
                    num_conv_block=num_conv_block_in_modeling,
                    dropout=dropout,
                    layer_dropout=layer_dropout,
                )
                for _ in range(num_modeling_encoder_block)
            ]
        )

        self.span_start_linear = nn.Linear(model_dim * 2, 1, bias=False)
        self.span_end_linear = nn.Linear(model_dim * 2, 1, bias=False)

        self.dropout = nn.Dropout(p=dropout)

        self.criterion = nn.CrossEntropyLoss()

    @overrides
    def forward(self, features, labels=None):
        """
            * Args:
                features: feature dictionary like below.
                    {"feature_name1": {
                         "token_name1": tensor,
                         "toekn_name2": tensor},
                     "feature_name2": ...}

            * Kwargs:
                label: label dictionary like below.
                    {"label_name1": tensor,
                     "label_name2": tensor}
                     Do not calculate loss when there is no label. (inference/predict mode)

            * Returns: output_dict (dict) consisting of
                - start_logits: representing unnormalized log probabilities of the span start position.
                - end_logits: representing unnormalized log probabilities of the span end position.
                - best_span: the string from the original passage that the model thinks is the best answer to the question.
                - answer_idx: the question id, mapping with answer
                - loss: A scalar loss to be optimised.
        """

        context = features["context"]
        question = features["question"]

        # 1. Input Embedding Layer
        query_params = {"frequent_word": {"frequent_tuning": True}}
        context_embed, query_embed = self.token_embedder(
            context, question, query_params=query_params, query_align=self.aligned_query_embedding
        )

        context_mask = f.get_mask_from_tokens(context).float()
        query_mask = f.get_mask_from_tokens(question).float()

        context_embed = self.context_highway(context_embed)
        context_embed = self.dropout(context_embed)
        context_embed = self.context_embed_pointwise_conv(context_embed)

        query_embed = self.query_highway(query_embed)
        query_embed = self.dropout(query_embed)
        query_embed = self.query_embed_pointwise_conv(query_embed)

        # 2. Embedding Encoder Layer
        for encoder_block in self.embed_encoder_blocks:
            context = encoder_block(context_embed)
            context_embed = context

            query = encoder_block(query_embed)
            query_embed = query

        # 3. Context-Query Attention Layer
        context_query_attention = self.co_attention(context, query, context_mask, query_mask)

        # Projection (memory issue)
        context_query_attention = self.pointwise_conv(context_query_attention)
        context_query_attention = self.dropout(context_query_attention)

        # 4. Model Encoder Layer
        model_encoder_block_inputs = context_query_attention

        # Stacked Model Encoder Block
        stacked_model_encoder_blocks = []
        for i in range(3):
            for _, model_encoder_block in enumerate(self.model_encoder_blocks):
                output = model_encoder_block(model_encoder_block_inputs, context_mask)
                model_encoder_block_inputs = output

            stacked_model_encoder_blocks.append(output)

        # 5. Output Layer
        span_start_inputs = torch.cat(
            [stacked_model_encoder_blocks[0], stacked_model_encoder_blocks[1]], dim=-1
        )
        span_start_inputs = self.dropout(span_start_inputs)
        span_start_logits = self.span_start_linear(span_start_inputs).squeeze(-1)

        span_end_inputs = torch.cat(
            [stacked_model_encoder_blocks[0], stacked_model_encoder_blocks[2]], dim=-1
        )
        span_end_inputs = self.dropout(span_end_inputs)
        span_end_logits = self.span_end_linear(span_end_inputs).squeeze(-1)

        # Masked Value
        span_start_logits = f.add_masked_value(span_start_logits, context_mask, value=-1e7)
        span_end_logits = f.add_masked_value(span_end_logits, context_mask, value=-1e7)

        output_dict = {
            "start_logits": span_start_logits,
            "end_logits": span_end_logits,
            "best_span": self.get_best_span(span_start_logits, span_end_logits),
        }

        if labels:
            answer_idx = labels["answer_idx"]
            answer_start_idx = labels["answer_start_idx"]
            answer_end_idx = labels["answer_end_idx"]

            output_dict["answer_idx"] = answer_idx

            # Loss
            loss = self.criterion(span_start_logits, answer_start_idx)
            loss += self.criterion(span_end_logits, answer_end_idx)
            output_dict["loss"] = loss.unsqueeze(0)  # NOTE: DataParallel concat Error

        return output_dict


class EncoderBlock(nn.Module):
    """
        Encoder Block

        []: residual
        position_encoding -> [convolution-layer] x # -> [self-attention-layer] -> [feed-forward-layer]

        - convolution-layer: depthwise separable convolutions
        - self-attention-layer: multi-head attention
        - feed-forward-layer: pointwise convolution

        * Args:
            model_dim: the number of model dimension
            num_heads: the number of head in multi-head attention
            kernel_size: convolution kernel size
            num_conv_block: the number of convolution block
            dropout: the dropout probability
            layer_dropout: the layer dropout probability
                (cf. Deep Networks with Stochastic Depth(https://arxiv.org/abs/1603.09382) )
    """

    def __init__(
        self,
        model_dim=128,
        num_head=8,
        kernel_size=5,
        num_conv_block=4,
        dropout=0.1,
        layer_dropout=0.9,
    ):
        super(EncoderBlock, self).__init__()

        self.position_encoding = encoder.PositionalEncoding(model_dim)
        self.dropout = nn.Dropout(dropout)

        self.num_conv_block = num_conv_block
        self.conv_blocks = nn.ModuleList(
            [conv.DepSepConv(model_dim, model_dim, kernel_size) for _ in range(num_conv_block)]
        )

        self.self_attention = attention.MultiHeadAttention(
            num_head=num_head, model_dim=model_dim, dropout=dropout
        )
        self.feedforward_layer = layer.PositionwiseFeedForward(
            model_dim, model_dim * 4, dropout=dropout
        )

        # survival probability for stochastic depth
        if layer_dropout < 1.0:
            L = (num_conv_block) + 2 - 1
            layer_dropout_prob = round(1 - (1 / L) * (1 - layer_dropout), 3)
            self.residuals = nn.ModuleList(
                layer.ResidualConnection(
                    model_dim, layer_dropout=layer_dropout_prob, layernorm=True
                )
                for l in range(num_conv_block + 2)
            )
        else:
            self.residuals = nn.ModuleList(
                layer.ResidualConnection(model_dim, layernorm=True)
                for l in range(num_conv_block + 2)
            )

    def forward(self, x, mask=None):
        # Positional Encoding
        x = self.position_encoding(x)

        # Convolution Block (LayerNorm -> Conv)
        for i, conv_block in enumerate(self.conv_blocks):
            x = self.residuals[i](x, sub_layer_fn=conv_block)
            x = self.dropout(x)

        # LayerNorm -> Self-attention
        self_attention = lambda x: self.self_attention(q=x, k=x, v=x, mask=mask)
        x = self.residuals[self.num_conv_block](x, sub_layer_fn=self_attention)
        x = self.dropout(x)

        # LayerNorm -> Feedforward layer
        x = self.residuals[self.num_conv_block + 1](x, sub_layer_fn=self.feedforward_layer)
        x = self.dropout(x)
        return x
