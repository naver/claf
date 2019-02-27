
from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F

from claf.modules.activation import get_activation_fn

from .base import TokenEmbedding


class CharEmbedding(TokenEmbedding):
    """
    Character Embedding (CharCNN)
    (https://arxiv.org/abs/1509.01626)

    * Args:
        vocab: Vocab (claf.tokens.vocab)

    * Kwargs:
        dropout: The number of dropout probability
        embed_dim: The number of embedding dimension
        kernel_sizes: The list of kernel size (n-gram)
        num_filter: The number of cnn filter
        activation: Activation Function (eg. ReLU)
    """

    def __init__(
        self, vocab, dropout=0.2, embed_dim=16, kernel_sizes=[5], num_filter=100, activation="relu"
    ):
        super(CharEmbedding, self).__init__(vocab)

        self.embed_dim = embed_dim
        self.num_filter = num_filter

        self.weight = self._init_weight(trainable=True)
        self.convs = [
            nn.Conv1d(
                in_channels=1,
                out_channels=num_filter,
                kernel_size=embed_dim * kernel_size,
                stride=embed_dim,
            )
            for kernel_size in kernel_sizes
        ]  # kernel_size = n-gram
        for i, conv in enumerate(self.convs):
            self.add_module(f"conv_{i}", conv)

        self.activation_fn = get_activation_fn(activation)()
        self.dropout = nn.Dropout(p=dropout)

        self.projection = None
        if len(kernel_sizes) > 1:
            maxpool_output_dim = len(kernel_sizes) * num_filter
            self.projection = nn.Linear(maxpool_output_dim, num_filter)

    def _init_weight(self, trainable=False):
        weight = torch.FloatTensor(self.get_vocab_size(), self.embed_dim)
        weight = torch.nn.Parameter(weight, requires_grad=trainable)
        torch.nn.init.xavier_uniform_(weight)
        return weight

    @overrides
    def forward(self, chars):
        mask_chars = (chars != 0).long()

        B, W_L, C_L = chars.size()  # (batch_size, word_maxlen, char_maxlen)
        chars = chars.view(B, W_L * C_L)

        char_embedds = F.embedding(chars, self.weight)
        char_embedds = char_embedds.view(B, W_L, C_L, -1)

        # Masking
        char_embedds = char_embedds * mask_chars.unsqueeze(-1).float()
        char_embedds = char_embedds.view(B * W_L, 1, -1)

        conv_outputs = []
        for i in range(len(self.convs)):
            conv = getattr(self, f"conv_{i}")
            output = self.activation_fn(conv(char_embedds))
            pooled = F.max_pool1d(output, output.size(2)).squeeze(2)

            conv_outputs.append(pooled)

        encoded = conv_outputs[0]
        if len(conv_outputs) > 1:
            encoded = torch.cat(conv_outputs, dim=1)
        encoded = encoded.view(B, W_L, -1)

        if self.projection:
            encoded = self.projection(encoded)
        return self.dropout(encoded)

    @overrides
    def get_output_dim(self):
        return self.num_filter
