from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F

from claf.decorator import register
from claf.model.base import ModelWithTokenEmbedder
from claf.model.reading_comprehension.mixin import SQuADv1
from claf.modules import attention, initializer
from claf.modules import functional as f


@register("model:docqa")
class DocQA(SQuADv1, ModelWithTokenEmbedder):
    """
    Document Reader Model. `Span Detector`

    Implementation of model presented in
    Simple and Effective Multi-Paragraph Reading Comprehension
    (https://arxiv.org/abs/1710.10723)

    - Embedding (Word + Char -> Contextual)
    - Attention
    - Residual self-attention
    - Output

    * Args:
        token_embedder: 'QATokenEmbedder', Used to embed the 'context' and 'question'.

    * Kwargs:
        lang_code: Dataset language code [en|ko]
        aligned_query_embedding: f_align(p_i) = sum(a_ij, E(qj), where the attention score a_ij
            captures the similarity between pi and each question words q_j.
            these features add soft alignments between similar but non-identical words (e.g., car and vehicle)
            it only apply to 'context_embed'.
        answer_maxlen: the most probable answer span of length less than or equal to {answer_maxlen}
        rnn_dim: the number of RNN cell dimension
        linear_dim: the number of attention linear dimension
        preprocess_rnn_num_layer: the number of recurrent layers (preprocess)
        modeling_rnn_num_layer: the number of recurrent layers (modeling)
        predict_rnn_num_layer: the number of recurrent layers (predict)
        dropout: the dropout probability
    """

    def __init__(
        self,
        token_embedder,
        lang_code="en",
        aligned_query_embedding=False,
        answer_maxlen=17,
        rnn_dim=100,
        linear_dim=200,
        preprocess_rnn_num_layer=1,
        modeling_rnn_num_layer=2,
        predict_rnn_num_layer=1,
        dropout=0.2,
        weight_init=True,
    ):
        super(DocQA, self).__init__(token_embedder)

        self.lang_code = lang_code
        self.aligned_query_embedding = aligned_query_embedding
        self.answer_maxlen = answer_maxlen
        self.token_embedder = token_embedder
        self.dropout = nn.Dropout(p=dropout)

        context_embed_dim, query_embed_dim = token_embedder.get_embed_dim()
        if self.aligned_query_embedding:
            context_embed_dim += query_embed_dim

        if context_embed_dim != query_embed_dim:
            self.context_preprocess_rnn = nn.GRU(
                input_size=context_embed_dim,
                hidden_size=rnn_dim,
                bidirectional=True,
                num_layers=preprocess_rnn_num_layer,
                batch_first=True,
            )
            self.query_preprocess_rnn = nn.GRU(
                input_size=query_embed_dim,
                hidden_size=rnn_dim,
                bidirectional=True,
                num_layers=preprocess_rnn_num_layer,
                batch_first=True,
            )
        else:
            preprocess_rnn = nn.GRU(
                input_size=context_embed_dim,
                hidden_size=rnn_dim,
                bidirectional=True,
                num_layers=preprocess_rnn_num_layer,
                batch_first=True,
            )

            self.context_preprocess_rnn = preprocess_rnn
            self.query_preprocess_rnn = preprocess_rnn

        self.bi_attention = attention.DocQAAttention(rnn_dim, linear_dim)
        self.attn_linear = nn.Linear(rnn_dim * 8, linear_dim)

        self.modeling_rnn = nn.GRU(
            input_size=linear_dim,
            hidden_size=rnn_dim,
            num_layers=modeling_rnn_num_layer,
            bidirectional=True,
            batch_first=True,
        )
        self.self_attention = SelfAttention(rnn_dim, linear_dim, weight_init=weight_init)

        self.span_start_rnn = nn.GRU(
            input_size=linear_dim,
            hidden_size=rnn_dim,
            bidirectional=True,
            num_layers=predict_rnn_num_layer,
            batch_first=True,
        )
        self.span_start_linear = nn.Linear(rnn_dim * 2, 1)

        self.span_end_rnn = nn.GRU(
            input_size=linear_dim + rnn_dim * 2,
            hidden_size=rnn_dim,
            bidirectional=True,
            num_layers=predict_rnn_num_layer,
            batch_first=True,
        )
        self.span_end_linear = nn.Linear(rnn_dim * 2, 1)

        self.activation_fn = F.relu
        self.criterion = nn.CrossEntropyLoss()

        if weight_init:
            modules = [
                self.context_preprocess_rnn,
                self.query_preprocess_rnn,
                self.modeling_rnn,
                self.attn_linear,
                self.span_start_rnn,
                self.span_start_linear,
                self.span_end_rnn,
                self.span_end_linear,
            ]
            initializer.weight(modules)

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

        # Sorted Sequence config (seq_lengths, perm_idx, unperm_idx) for RNN pack_forward
        context_seq_config = f.get_sorted_seq_config(context)
        query_seq_config = f.get_sorted_seq_config(question)

        # Embedding
        query_params = {"frequent_word": {"frequent_tuning": True}}
        context_embed, query_embed = self.token_embedder(
            context, question, query_params=query_params, query_align=self.aligned_query_embedding
        )

        context_mask = f.get_mask_from_tokens(context).float()  # B X 1 X C_L
        query_mask = f.get_mask_from_tokens(question).float()  # B X 1 X Q_L

        # Pre-process
        context_embed = self.dropout(context_embed)
        context_encoded = f.forward_rnn_with_pack(
            self.context_preprocess_rnn, context_embed, context_seq_config
        )
        context_encoded = self.dropout(context_encoded)

        query_embed = self.dropout(query_embed)
        query_encoded = f.forward_rnn_with_pack(
            self.query_preprocess_rnn, query_embed, query_seq_config
        )
        query_encoded = self.dropout(query_encoded)

        # Attention -> Projection
        context_attnded = self.bi_attention(
            context_encoded, context_mask, query_encoded, query_mask
        )
        context_attnded = self.activation_fn(self.attn_linear(context_attnded))  # B X C_L X dim*2

        # Residual Self-Attention
        context_attnded = self.dropout(context_attnded)
        context_encoded = f.forward_rnn_with_pack(
            self.modeling_rnn, context_attnded, context_seq_config
        )
        context_encoded = self.dropout(context_encoded)

        context_self_attnded = self.self_attention(context_encoded, context_mask)  # B X C_L X dim*2
        context_final = self.dropout(context_attnded + context_self_attnded)  # B X C_L X dim*2

        # Prediction
        span_start_input = f.forward_rnn_with_pack(
            self.span_start_rnn, context_final, context_seq_config
        )  # B X C_L X dim*2
        span_start_input = self.dropout(span_start_input)
        span_start_logits = self.span_start_linear(span_start_input).squeeze(-1)  # B X C_L

        span_end_input = torch.cat([span_start_input, context_final], dim=-1)  # B X C_L X dim*4
        span_end_input = f.forward_rnn_with_pack(
            self.span_end_rnn, span_end_input, context_seq_config
        )  # B X C_L X dim*2
        span_end_input = self.dropout(span_end_input)
        span_end_logits = self.span_end_linear(span_end_input).squeeze(-1)  # B X C_L

        # Masked Value
        span_start_logits = f.add_masked_value(span_start_logits, context_mask, value=-1e7)
        span_end_logits = f.add_masked_value(span_end_logits, context_mask, value=-1e7)

        output_dict = {
            "start_logits": span_start_logits,
            "end_logits": span_end_logits,
            "best_span": self.get_best_span(
                span_start_logits, span_end_logits, answer_maxlen=self.answer_maxlen
            ),
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


class SelfAttention(nn.Module):
    """
        Same bi-attention mechanism, only now between the passage and itself.
    """

    def __init__(self, rnn_dim, linear_dim, dropout=0.2, weight_init=True):
        super(SelfAttention, self).__init__()

        self.self_attention = attention.DocQAAttention(
            rnn_dim, linear_dim, self_attn=True, weight_init=weight_init
        )
        self.self_attn_Linear = nn.Linear(rnn_dim * 6, linear_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.activation_fn = F.relu

        if weight_init:
            initializer.weight(self.self_attn_Linear)

    def forward(self, context, context_mask):
        context_self_attnded = self.self_attention(context, context_mask, context, context_mask)
        return self.activation_fn(self.self_attn_Linear(context_self_attnded))
