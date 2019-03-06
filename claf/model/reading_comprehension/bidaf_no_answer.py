
from overrides import overrides
import torch
import torch.nn as nn

from claf.decorator import register
from claf.model.base import ModelWithTokenEmbedder
from claf.model.reading_comprehension.mixin import SQuADv2
import claf.modules.functional as f
import claf.modules.attention as attention
import claf.modules.layer as layer


@register("model:bidaf_no_answer")
class BiDAF_No_Answer(SQuADv2, ModelWithTokenEmbedder):
    """
    Question Answering Model. `Span Detector`, `No Answer`

    Bidirectional Attention Flow for Machine Comprehension + Bias (No_Answer)

    - Embedding (Word + Char -> Contextual)
    - Attention Flow
    - Modeling (RNN)
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
        model_dim: the number of model dimension
        dropout: the dropout probability
    """

    def __init__(
        self,
        token_embedder,
        lang_code="en",
        aligned_query_embedding=False,
        answer_maxlen=None,
        model_dim=100,
        contextual_rnn_num_layer=1,
        modeling_rnn_num_layer=2,
        predict_rnn_num_layer=1,
        dropout=0.2,
    ):
        super(BiDAF_No_Answer, self).__init__(token_embedder)

        self.lang_code = lang_code
        self.aligned_query_embedding = aligned_query_embedding
        self.answer_maxlen = answer_maxlen
        self.token_embedder = token_embedder
        self.dropout = nn.Dropout(p=dropout)

        context_embed_dim, query_embed_dim = token_embedder.get_embed_dim()
        if self.aligned_query_embedding:
            context_embed_dim += query_embed_dim

        if context_embed_dim != query_embed_dim:
            self.context_highway = layer.Highway(context_embed_dim)
            self.context_contextual_rnn = nn.LSTM(
                input_size=context_embed_dim,
                hidden_size=model_dim,
                bidirectional=True,
                num_layers=contextual_rnn_num_layer,
                batch_first=True,
            )

            self.query_highway = layer.Highway(query_embed_dim)
            self.query_contextual_rnn = nn.LSTM(
                input_size=query_embed_dim,
                hidden_size=model_dim,
                bidirectional=True,
                num_layers=contextual_rnn_num_layer,
                batch_first=True,
            )
        else:
            highway = layer.Highway(query_embed_dim)

            self.context_highway = highway
            self.query_highway = highway

            contextual_rnn = nn.LSTM(
                input_size=context_embed_dim,
                hidden_size=model_dim,
                bidirectional=True,
                num_layers=contextual_rnn_num_layer,
                batch_first=True,
            )

            self.context_contextual_rnn = contextual_rnn
            self.query_contextual_rnn = contextual_rnn

        self.attention = attention.BiAttention(model_dim)
        self.modeling_rnn = nn.LSTM(
            input_size=8 * model_dim,
            hidden_size=model_dim,
            num_layers=modeling_rnn_num_layer,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
        )
        self.output_end_rnn = nn.LSTM(
            input_size=14 * model_dim,
            hidden_size=model_dim,
            bidirectional=True,
            num_layers=predict_rnn_num_layer,
            batch_first=True,
        )

        self.span_start_linear = nn.Linear(10 * model_dim, 1)
        self.span_end_linear = nn.Linear(10 * model_dim, 1)

        self.bias = nn.Parameter(torch.randn(1, 1))

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

        # Sorted Sequence config (seq_lengths, perm_idx, unperm_idx) for RNN pack_forward
        context_seq_config = f.get_sorted_seq_config(context)
        query_seq_config = f.get_sorted_seq_config(question)

        # Embedding Layer (Char + Word -> Contextual)
        query_params = {"frequent_word": {"frequent_tuning": True}}
        context_embed, query_embed = self.token_embedder(
            context, question, query_params=query_params, query_align=self.aligned_query_embedding
        )

        context_mask = f.get_mask_from_tokens(context).float()
        query_mask = f.get_mask_from_tokens(question).float()

        B, C_L = context_embed.size(0), context_embed.size(1)

        context_embed = self.context_highway(context_embed)
        query_embed = self.query_highway(query_embed)

        context_encoded = f.forward_rnn_with_pack(
            self.context_contextual_rnn, context_embed, context_seq_config
        )
        context_encoded = self.dropout(context_encoded)

        query_encoded = f.forward_rnn_with_pack(
            self.query_contextual_rnn, query_embed, query_seq_config
        )
        query_encoded = self.dropout(query_encoded)

        # Attention Flow Layer
        attention_context_query = self.attention(
            context_encoded, context_mask, query_encoded, query_mask
        )

        # Modeling Layer
        modeled_context = f.forward_rnn_with_pack(
            self.modeling_rnn, attention_context_query, context_seq_config
        )
        modeled_context = self.dropout(modeled_context)

        M_D = modeled_context.size(-1)

        # Output Layer
        span_start_input = self.dropout(
            torch.cat([attention_context_query, modeled_context], dim=-1)
        )  # (B, C_L, 10d)
        span_start_logits = self.span_start_linear(span_start_input).squeeze(-1)  # (B, C_L)
        span_start_probs = f.masked_softmax(span_start_logits, context_mask)

        span_start_representation = f.weighted_sum(
            attention=span_start_probs, matrix=modeled_context
        )
        tiled_span_start_representation = span_start_representation.unsqueeze(1).expand(B, C_L, M_D)

        span_end_representation = torch.cat(
            [
                attention_context_query,
                modeled_context,
                tiled_span_start_representation,
                modeled_context * tiled_span_start_representation,
            ],
            dim=-1,
        )
        encoded_span_end = f.forward_rnn_with_pack(
            self.output_end_rnn, span_end_representation, context_seq_config
        )
        encoded_span_end = self.dropout(encoded_span_end)

        span_end_input = self.dropout(
            torch.cat([attention_context_query, encoded_span_end], dim=-1)
        )
        span_end_logits = self.span_end_linear(span_end_input).squeeze(-1)

        # Masked Value
        span_start_logits = f.add_masked_value(span_start_logits, context_mask, value=-1e7)
        span_end_logits = f.add_masked_value(span_end_logits, context_mask, value=-1e7)

        # No_Answer Bias
        bias = self.bias.expand(B, 1)
        span_start_logits = torch.cat([span_start_logits, bias], dim=-1)
        span_end_logits = torch.cat([span_end_logits, bias], dim=-1)

        output_dict = {
            "start_logits": span_start_logits,
            "end_logits": span_end_logits,
            "best_span": self.get_best_span(
                span_start_logits[:, :-1],
                span_end_logits[:, :-1],
                answer_maxlen=self.answer_maxlen,  # except no_answer bias
            ),
        }

        if labels:
            answer_idx = labels["answer_idx"]
            answer_start_idx = labels["answer_start_idx"]
            answer_end_idx = labels["answer_end_idx"]
            answerable = labels["answerable"]

            # No_Asnwer Case
            C_L = context_mask.size(1)
            answer_start_idx = answer_start_idx.masked_fill(answerable.eq(0), C_L)
            answer_end_idx = answer_end_idx.masked_fill(answerable.eq(0), C_L)

            output_dict["answer_idx"] = answer_idx

            # Loss
            loss = self.criterion(span_start_logits, answer_start_idx)
            loss += self.criterion(span_end_logits, answer_end_idx)
            output_dict["loss"] = loss.unsqueeze(0)  # NOTE: DataParallel concat Error

        return output_dict
