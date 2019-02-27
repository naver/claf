
from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F

from claf.tokens.vocabulary import Vocab

from .base import TokenEmbedding
from .word_embedding import WordEmbedding


class SparseFeature(TokenEmbedding):
    """
    Sparse Feature

    1. Sparse to Embedding
    2. One Hot Encoding

    * Args:
        vocab: Vocab (claf.tokens.vocab)
        embed_type: The type of embedding [one_hot|embedding]
        feature_count: The number of feature count

    * Kwargs:
        params: additional parameters for embedding module
    """

    def __init__(self, vocab, embed_type, feature_count, params={}):
        super(SparseFeature, self).__init__(vocab)

        self.feature_count = feature_count

        if embed_type == "embedding":
            embed_module = SparseToEmbedding
        else:
            embed_module = OneHotEncoding

        self.embed_modules = nn.ModuleList(
            [embed_module(i, vocab.token_name, **params) for i in range(feature_count)]
        )

        indexs = torch.arange(feature_count).long()
        indexs = indexs.view(feature_count, 1)
        self.indexs = nn.Parameter(indexs, requires_grad=False)

    @overrides
    def forward(self, inputs):
        embedded_inputs = []

        for i in range(len(self.embed_modules)):
            tensors = torch.index_select(inputs, -1, self.indexs[i]).squeeze(-1)
            embedded = self.embed_modules[i](tensors)

            embedded_inputs.append(embedded)
        return torch.cat(embedded_inputs, dim=-1)

    @overrides
    def get_output_dim(self):
        return sum(e.get_output_dim() for e in self.embed_modules)


class SparseToEmbedding(nn.Module):
    """
    Sparse to Embedding

    * Args:
        token_name: token_name

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
        index,
        token_name,
        classes,
        dropout=0,
        embed_dim=15,
        trainable=True,
        padding_idx=None,
        max_norm=None,
        norm_type=2,
        scale_grad_by_freq=False,
        sparse=False,
    ):
        super(SparseToEmbedding, self).__init__()

        self.embed_dim = embed_dim

        vocab = Vocab(token_name)
        vocab.init()
        for c in classes[index]:
            vocab.add(c)

        embedding_params = {
            "vocab": vocab,
            "dropout": dropout,
            "embed_dim": embed_dim,
            "trainable": trainable,
            "padding_idx": padding_idx,
            "max_norm": max_norm,
            "norm_type": norm_type,
            "scale_grad_by_freq": scale_grad_by_freq,
            "sparse": sparse,
        }

        self.embedding = WordEmbedding(**embedding_params)

    @overrides
    def forward(self, inputs):
        return self.embedding(inputs)

    def get_output_dim(self):
        return self.embed_dim


class OneHotEncoding(nn.Module):
    """
    Sparse to one-hot encoding

    * Args:
        vocab: Vocab (claf.tokens.vocab)

    """

    def __init__(self, index, token_name, classes):
        super(OneHotEncoding, self).__init__()

        vocab = Vocab(token_name)
        vocab.init()
        for c in classes[index]:
            vocab.add(c)

        num_class = len(vocab)
        self.num_class = num_class

        one_hot_encoding = torch.eye(num_class)
        self.one_hots = nn.Parameter(one_hot_encoding, requires_grad=False)

    @overrides
    def forward(self, inputs):
        if self.num_class == 4:
            inputs = inputs - 2  # make 0, 1 binary_feature
            return inputs.float().unsqueeze(-1)

        return F.embedding(inputs, self.one_hots)

    def get_output_dim(self):
        if self.num_class == 4:  # binary_feature
            return 1  # 0 or 1

        return self.num_class
