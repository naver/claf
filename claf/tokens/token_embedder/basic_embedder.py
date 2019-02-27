
from overrides import overrides

import torch

from .base import TokenEmbedder


class BasicTokenEmbedder(TokenEmbedder):
    """
    Basic Token Embedder

    Take a tensor(indexed token) look up Embedding modules.
    Output is concatenating all embedded tensors.

    * Args:
        token_makers: dictionary of TokenMaker (claf.tokens.token_maker)
    """

    def __init__(self, token_makers):
        super(BasicTokenEmbedder, self).__init__(token_makers)

    @overrides
    def get_embed_dim(self, except_keys=[]):
        return sum(self.embed_dims.values())

    @overrides
    def forward(self, inputs, except_keys=[], params={}):
        token_names = [name for name in self.token_names if name not in except_keys]
        if set(token_names) != set(inputs.keys()):
            raise ValueError(
                f"Mismatch token_names  inputs: {inputs.keys()}, embeddings: {self.token_names}"
            )

        embedded_tokens = []
        for token_name, tensors in inputs.items():
            embedding = getattr(self, token_name)

            embedded_token = embedding(tensors, **params)
            embedded_tokens.append(embedded_token)

        output = torch.cat(embedded_tokens, dim=-1)
        return output
