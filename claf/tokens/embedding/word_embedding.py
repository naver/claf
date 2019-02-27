
import logging
from overrides import overrides
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from claf.data.data_handler import CachePath, DataHandler

from .base import TokenEmbedding

logger = logging.getLogger(__name__)


class WordEmbedding(TokenEmbedding):
    """
    Word Embedding
    Default Token Embedding

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
        trainable=True,
    ):
        super(WordEmbedding, self).__init__(vocab)
        self.data_handler = DataHandler(cache_path=CachePath.PRETRAINED_VECTOR)

        self.embed_dim = embed_dim
        if dropout and dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x

        if pretrained_path:
            weight = self._read_pretrained_file(pretrained_path)
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)
        else:
            self.weight = self._init_weight(trainable=trainable)

        # nn.functional.embedding = optional paramters
        #  (padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
        # check - https://pytorch.org/docs/master/nn.html#torch.nn.functional.embeddin\
        #    ://pytorch.org/docs/master/nn.html#torch.nn.functional.embedding
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

    def _init_weight(self, trainable=True):
        weight = torch.FloatTensor(self.get_vocab_size(), self.embed_dim)
        weight = torch.nn.Parameter(weight, requires_grad=trainable)
        torch.nn.init.xavier_uniform_(weight)
        return weight

    @overrides
    def forward(self, words):
        input_size = words.size()
        if len(input_size) > 2:
            words = words.view(-1, input_size[-1])

        embedded_words = F.embedding(
            words,
            self.weight,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

        if len(input_size) > 2:
            embedded_size = list(input_size) + [embedded_words.size(-1)]
            embedded_words = embedded_words.view(*embedded_size)
        return self.dropout(embedded_words)

    def _read_pretrained_file(self, file_path):
        words_to_keep = set(self.vocab.get_all_tokens())
        vocab_size = self.get_vocab_size()
        embeddings = {}

        # First we read the embeddings from the file, only keeping vectors for the words we need.
        logger.info("Reading embeddings from file")
        file_path = self.data_handler.read(file_path, return_path=True)
        with open(file_path, "rb") as embeddings_file:
            for line in embeddings_file:
                fields = line.decode("utf-8").rstrip().split(" ")

                if len(fields) - 1 != self.embed_dim:
                    logger.info(
                        f"Found line with wrong number of dimensions (expected {self.embed_dim}, was {len(fields)}): {line}"
                    )
                    continue

                word = fields[0]
                if word in words_to_keep:
                    vector = np.asarray(fields[1:], dtype="float32")
                    embeddings[word] = vector

        if not embeddings:
            raise ValueError(
                "No embeddings of correct dimension found. check input dimension value"
            )

        all_embeddings = np.asarray(list(embeddings.values()))
        embeddings_mean = float(np.mean(all_embeddings))
        embeddings_std = float(np.std(all_embeddings))
        # Now we initialize the weight matrix for an embedding layer, starting with random vectors,
        # then filling in the word vectors we just read.
        logger.info("Initializing pre-trained embedding layer")
        embedding_matrix = torch.FloatTensor(vocab_size, self.embed_dim).normal_(
            embeddings_mean, embeddings_std
        )

        match_count = 0
        for i in range(0, vocab_size):
            word = self.vocab.get_token(i)
            if word in embeddings:
                embedding_matrix[i] = torch.FloatTensor(embeddings[word])
                match_count += 1
            else:
                # f"Word {word} was not found in the embedding file. Initialising randomly."
                pass
        logger.info(f"Match embedding vocab size: {match_count}.  [{match_count}/{vocab_size}]")
        return embedding_matrix

    @overrides
    def get_output_dim(self):
        return self.embed_dim
