from overrides import overrides

import torch.nn as nn

from claf.decorator import register
from claf.model.base import ModelWithTokenEmbedder
from claf.model.reading_comprehension.mixin import SQuADv1
import claf.modules.functional as f
import claf.modules.attention as attention


@register("model:drqa")
class DrQA(SQuADv1, ModelWithTokenEmbedder):
    """
    Document Reader Model. `Span Detector`

    Implementation of model presented in
    Reading Wikipedia to Answer Open-Domain Questions
    (https://arxiv.org/abs/1704.00051)

    - Embedding + features
    - Align question embedding

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
        dropout: the dropout probability
    """

    def __init__(
        self,
        token_embedder,
        lang_code="en",
        aligned_query_embedding=False,
        answer_maxlen=None,
        model_dim=128,
        dropout=0.3,
    ):
        super(DrQA, self).__init__(token_embedder)

        self.lang_code = lang_code
        self.aligned_query_embedding = aligned_query_embedding
        self.answer_maxlen = answer_maxlen
        self.token_embedder = token_embedder
        self.dropout = nn.Dropout(p=dropout)

        context_embed_dim, query_embed_dim = token_embedder.get_embed_dim()
        if self.aligned_query_embedding:
            context_embed_dim += query_embed_dim

        self.paragraph_rnn = nn.LSTM(
            input_size=context_embed_dim,
            hidden_size=model_dim,
            num_layers=3,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )

        self.query_rnn = nn.LSTM(
            input_size=query_embed_dim,
            hidden_size=model_dim,
            num_layers=3,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )

        self.query_att = attention.LinearSeqAttn(model_dim * 2)

        self.start_attn = attention.BilinearSeqAttn(model_dim * 2, model_dim * 2)
        self.end_attn = attention.BilinearSeqAttn(model_dim * 2, model_dim * 2)

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

        context = features["context"]  # aka paragraph
        question = features["question"]

        # Sorted Sequence config (seq_lengths, perm_idx, unperm_idx) for RNN pack_forward
        context_seq_config = f.get_sorted_seq_config(context)
        query_seq_config = f.get_sorted_seq_config(question)

        # Embedding
        query_params = {"frequent_word": {"frequent_tuning": True}}
        context_embed, query_embed = self.token_embedder(
            context, question, query_params=query_params, query_align=self.aligned_query_embedding
        )

        context_mask = f.get_mask_from_tokens(context).float()
        query_mask = f.get_mask_from_tokens(question).float()

        context_embed = self.dropout(context_embed)
        query_embed = self.dropout(query_embed)

        # RNN (LSTM)
        context_encoded = f.forward_rnn_with_pack(
            self.paragraph_rnn, context_embed, context_seq_config
        )
        context_encoded = self.dropout(context_encoded)

        query_encoded = f.forward_rnn_with_pack(
            self.query_rnn, query_embed, query_seq_config
        )  # (B, Q_L, H*2)
        query_encoded = self.dropout(query_encoded)

        query_attention = self.query_att(query_encoded, query_mask)  # (B, Q_L)
        query_att_sum = f.weighted_sum(query_attention, query_encoded)  # (B, H*2)

        span_start_logits = self.start_attn(context_encoded, query_att_sum, context_mask)
        span_end_logits = self.end_attn(context_encoded, query_att_sum, context_mask)

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

            loss = self.criterion(span_start_logits, answer_start_idx)
            loss += self.criterion(span_end_logits, answer_end_idx)
            output_dict["loss"] = loss.unsqueeze(0)

        return output_dict
