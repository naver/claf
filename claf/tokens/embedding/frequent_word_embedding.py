
from overrides import overrides
import torch
import torch.nn as nn

import claf.modules.functional as f

from .base import TokenEmbedding
from .word_embedding import WordEmbedding


class FrequentTuningWordEmbedding(TokenEmbedding):
    """
    Frequent Word Finetuning Embedding
    Finetuning embedding matrix, according to 'threshold_index'

    * Args:
        vocab: Vocab (claf.tokens.vocab)

    * Kwargs:
        dropout: The number of dropout probability
        embed_dim: The number of embedding dimension
        padding_idx: If given, pads the output with the embedding vector at padding_idx
            (initialized to zeros) whenever it encounters the index.
        max_norm: If given, will renormalize the embedding vectors to have a norm lesser
            than this before extracting. Note: this will modify weight in-place.
        norm_type: The p of the p-norm to compute for the max_norm option. Default 2.
        scale_grad_by_freq: if given, this will scale gradients by the inverse of
            frequency of the words in the mini-batch. Default False.
        sparse: if True, gradient w.r.t. weight will be a sparse tensor.
            See Notes under torch.nn.Embedding for more details regarding sparse gradients.
        pretrained_path: pretrained vector path (eg. GloVe)
        trainable: finetune or fixed
    """

    def __init__(
        self,
        vocab,
        dropout=0.2,
        embed_dim=100,
        padding_idx=None,
        max_norm=None,
        norm_type=2,
        scale_grad_by_freq=False,
        sparse=False,
        pretrained_path=None,
    ):
        super(FrequentTuningWordEmbedding, self).__init__(vocab)

        self.threshold_index = vocab.threshold_index

        self.embed_dim = embed_dim
        self.fine_tune_word_embedding = WordEmbedding(
            vocab,
            dropout=0,
            embed_dim=embed_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            pretrained_path=pretrained_path,
        )
        self.fixed_word_embedding = WordEmbedding(
            vocab,
            dropout=0,
            embed_dim=embed_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            pretrained_path=pretrained_path,
        )

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x

    @overrides
    def forward(self, words, frequent_tuning=False):
        if frequent_tuning and self.training:

            padding_mask = words.eq(0).long()

            # Fine-tuning - N the most frequent
            fine_tune_mask = torch.lt(words, self.threshold_index) * padding_mask.eq(
                0
            )  # < threshold_index
            fine_tune_words = words * fine_tune_mask.long()

            fine_tune_embedded = self.fine_tune_word_embedding(fine_tune_words)
            fine_tune_embedded = f.masked_zero(fine_tune_embedded, fine_tune_mask)

            # Fixed - under N frequent
            fixed_mask = torch.ge(words, self.threshold_index)  # >= threshold_index

            fixed_embedeed = self.fixed_word_embedding(words).detach()  # Fixed
            fixed_embedeed = f.masked_zero(fixed_embedeed, fixed_mask)

            embedded_words = fine_tune_embedded + fixed_embedeed
        else:
            embedded_words = self.fixed_word_embedding(words)

        return self.dropout(embedded_words)

    @overrides
    def get_output_dim(self):
        return self.embed_dim
