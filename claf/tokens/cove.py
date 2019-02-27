"""
This code is from salesforce/cove
(https://github.com/salesforce/cove/blob/master/cove/encoder.py)
"""

import torch
from torch import nn

from claf.data.data_handler import CachePath, DataHandler


class MTLSTM(nn.Module):
    def __init__(
        self, word_embedding, pretrained_path=None, requires_grad=False, residual_embeddings=False
    ):
        """Initialize an MTLSTM.

        Arguments:
            n_vocab (bool): If not None, initialize MTLSTM with an embedding matrix with n_vocab vectors
            vectors (Float Tensor): If not None, initiapize embedding matrix with specified vectors
            residual_embedding (bool): If True, concatenate the input embeddings with MTLSTM outputs during forward
        """
        super(MTLSTM, self).__init__()
        self.word_embedding = word_embedding
        self.rnn = nn.LSTM(300, 300, num_layers=2, bidirectional=True, batch_first=True)

        data_handler = DataHandler(cache_path=CachePath.PRETRAINED_VECTOR)
        cove_weight_path = data_handler.read(pretrained_path, return_path=True)

        if torch.cuda.is_available():
            checkpoint = torch.load(cove_weight_path)
        else:
            checkpoint = torch.load(cove_weight_path, map_location="cpu")

        self.rnn.load_state_dict(checkpoint)
        self.residual_embeddings = residual_embeddings
        self.requires_grad = requires_grad

    def forward(self, inputs):
        """A pretrained MT-LSTM (McCann et. al. 2017).
        This LSTM was trained with 300d 840B GloVe on the WMT 2017 machine translation dataset.

        Arguments:
            inputs (Tensor): If MTLSTM handles embedding, a Long Tensor of size (batch_size, timesteps).
                             Otherwise, a Float Tensor of size (batch_size, timesteps, features).
            lengths (Long Tensor): (batch_size, lengths) lenghts of each sequence for handling padding
            hidden (Float Tensor): initial hidden state of the LSTM
        """
        embedded_inputs = self.word_embedding(inputs)
        encoded_inputs, _ = self.rnn(embedded_inputs)
        if not self.requires_grad:
            encoded_inputs.detach()

        outputs = encoded_inputs
        if self.residual_embeddings:
            outputs = torch.cat([embedded_inputs, encoded_inputs], 2)

        return outputs
