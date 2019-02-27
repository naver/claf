

import torch


class TokenEmbedder(torch.nn.Module):
    """
    Token Embedder

    Take a tensor(indexed token) look up Embedding modules.

    * Args:
        token_makers: dictionary of TokenMaker (claf.token_makers.token)
    """

    def __init__(self, token_makers):
        super(TokenEmbedder, self).__init__()

        self.embed_dims = {}

        self.vocabs = {
            token_name: token_maker.vocab for token_name, token_maker in token_makers.items()
        }
        self.add_embedding_modules(token_makers)

    def add_embedding_modules(self, token_makers):
        """ add embedding module to TokenEmbedder """
        self.token_names = []
        for token_name, token_maker in token_makers.items():
            self.token_names.append(token_name)

            vocab = token_maker.vocab
            embedding = token_maker.embedding_fn(vocab)
            self.add_module(token_name, embedding)

            self.embed_dims[token_name] = embedding.get_output_dim()

    def get_embed_dim(self):
        raise NotImplementedError

    def forward(self, inputs, params={}):
        raise NotImplementedError
