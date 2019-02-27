
import torch


class TokenEmbedding(torch.nn.Module):
    """
    Token Embedding

    It can be embedding matrix, language model (ELMo), neural machine translation model (CoVe) and features.

    * Args:
        vocab: Vocab (rqa.tokens.vocab)
    """

    def __init__(self, vocab):
        super(TokenEmbedding, self).__init__()

        self.vocab = vocab

    def forward(self, tokens):
        """ embedding look-up """
        raise NotImplementedError

    def get_output_dim(self):
        """ get embedding dimension """
        raise NotImplementedError

    def get_vocab_size(self):
        return len(self.vocab)
