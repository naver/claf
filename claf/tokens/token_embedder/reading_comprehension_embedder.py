
from overrides import overrides
import torch

import claf.modules.functional as f
import claf.modules.attention as attention

from .base import TokenEmbedder


class RCTokenEmbedder(TokenEmbedder):
    """
    Reading Comprehension Token Embedder

    Take a tensor(indexed token) look up Embedding modules.
    Inputs are seperated context and query for individual token setting.

    * Args:
        token_makers: dictionary of TokenMaker (claf.tokens.token_maker)
        vocabs: dictionary of vocab
            {"token_name": Vocab (claf.token_makers.vocaburary), ...}
    """

    EXCLUSIVE_TOKENS = ["exact_match"]  # only context

    def __init__(self, token_makers):
        super(RCTokenEmbedder, self).__init__(token_makers)

        self.context_embed_dim = sum(self.embed_dims.values())
        self.query_embed_dim = sum(self._filter(self.embed_dims, exclusive=False).values())

        self.align_attention = attention.SeqAttnMatch(self.query_embed_dim)

    @overrides
    def get_embed_dim(self):
        return self.context_embed_dim, self.query_embed_dim

    @overrides
    def forward(self, context, query, context_params={}, query_params={}, query_align=False):
        """
        * Args:
            context: context inputs (eg. {"token_name1": tensor, "token_name2": tensor, ...})
            query: query inputs (eg. {"token_name1": tensor, "token_name2": tensor, ...})

        * Kwargs:
            context_params: custom context parameters
            query_params: query context parameters
            query_align: f_align(p_i) = sum(a_ij, E(qj), where the attention score a_ij
                captures the similarity between pi and each question words q_j.
                these features add soft alignments between similar but non-identical words (e.g., car and vehicle)
                it only apply to 'context_embed'.
        """

        if set(self.token_names) != set(context.keys()):
            raise ValueError(
                f"Mismatch token_names  inputs: {context.keys()}, embeddings: {self.token_names}"
            )

        context_tokens, query_tokens = {}, {}
        for token_name, context_tensors in context.items():
            embedding = getattr(self, token_name)

            context_tokens[token_name] = embedding(
                context_tensors, **context_params.get(token_name, {})
            )
            if token_name in query:
                query_tokens[token_name] = embedding(
                    query[token_name], **query_params.get(token_name, {})
                )

        # query_align_embedding
        if query_align:
            common_context = self._filter(context_tokens, exclusive=False)
            embedded_common_context = torch.cat(list(common_context.values()), dim=-1)
            exclusive_context = self._filter(context_tokens, exclusive=True)

            embedded_exclusive_context = None
            if exclusive_context != {}:
                embedded_exclusive_context = torch.cat(list(exclusive_context.values()), dim=-1)

            query_mask = f.get_mask_from_tokens(query_tokens)
            embedded_query = torch.cat(list(query_tokens.values()), dim=-1)

            embedded_aligned_query = self.align_attention(
                embedded_common_context, embedded_query, query_mask
            )

            # Merge context embedded
            embedded_context = [embedded_common_context, embedded_aligned_query]
            if embedded_exclusive_context is not None:
                embedded_context.append(embedded_exclusive_context)

            context_output = torch.cat(embedded_context, dim=-1)
            query_output = embedded_query
        else:
            context_output = torch.cat(list(context_tokens.values()), dim=-1)
            query_output = torch.cat(list(query_tokens.values()), dim=-1)

        return context_output, query_output

    def _filter(self, token_data, exclusive=False):
        if exclusive:
            return {k: v for k, v in token_data.items() if k in self.EXCLUSIVE_TOKENS}
        else:
            return {k: v for k, v in token_data.items() if k not in self.EXCLUSIVE_TOKENS}
