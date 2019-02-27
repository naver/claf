

from overrides import overrides

import torch.nn as nn

from claf.tokens.cove import MTLSTM

from .base import TokenEmbedding
from .word_embedding import WordEmbedding


class CoveEmbedding(TokenEmbedding):
    """
    Cove Embedding

    Learned in Translation: Contextualized Word Vectors
    (http://papers.nips.cc/paper/7209-learned-in-translation-contextualized-word-vectors.pdf)

    * Args:
        vocab: Vocab (claf.tokens.vocab)

    * Kwargs:
        dropout: The number of dropout probability
        pretrained_path: pretrained vector path (eg. GloVe)
        trainable: finetune or fixed
        project_dim: The number of project (linear) dimension
    """

    def __init__(
        self,
        vocab,
        glove_pretrained_path=None,
        model_pretrained_path=None,
        dropout=0.2,
        trainable=False,
        project_dim=None,
    ):
        super(CoveEmbedding, self).__init__(vocab)

        self.embed_dim = 600  # MTLSTM (hidden_size=300 + bidirectional => 600)
        word_embedding = WordEmbedding(
            vocab, dropout=0, embed_dim=300, pretrained_path=glove_pretrained_path
        )
        self.cove = MTLSTM(
            word_embedding, pretrained_path=model_pretrained_path, requires_grad=trainable
        )

        if dropout and dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x

        self.project_dim = project_dim
        self.project_linear = None
        if project_dim:
            self.project_linear = nn.Linear(self.elmo.get_output_dim(), project_dim)

    @overrides
    def forward(self, words):
        embedded_words = self.cove(words)
        return self.dropout(embedded_words)

    @overrides
    def get_output_dim(self):
        if self.project_linear:
            return self.project_dim
        return self.embed_dim
