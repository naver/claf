
from overrides import overrides

import torch.nn as nn

from claf.data.data_handler import CachePath, DataHandler
from claf.tokens.elmo import Elmo

from .base import TokenEmbedding


DEFAULT_OPTIONS_FILE = "elmo_2x4096_512_2048cnn_2xhighway_options.json"
DEFAULT_WEIGHT_FILE = "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
HIDDEN_SIZE = 1024


class ELMoEmbedding(TokenEmbedding):
    """
    ELMo Embedding
    Embedding From Language Model

    Deep contextualized word representations
    (https://arxiv.org/abs/1802.0536)

    * Args:
        vocab: Vocab (claf.tokens.vocab)

    * Kwargs:
        options_file: ELMo model config file path
        weight_file: ELMo model weight file path
        do_layer_norm: Should we apply layer normalization (passed to ``ScalarMix``)?
            default is False
        dropout: The number of dropout probability
        trainable: Finetune or fixed
        project_dim: The number of project (linear) dimension
    """

    def __init__(
        self,
        vocab,
        options_file=DEFAULT_OPTIONS_FILE,
        weight_file=DEFAULT_WEIGHT_FILE,
        do_layer_norm=False,
        dropout=0.5,
        trainable=False,
        project_dim=None,
    ):
        super(ELMoEmbedding, self).__init__(vocab)
        data_handler = DataHandler(cache_path=CachePath.PRETRAINED_VECTOR)
        option_path = data_handler.read(options_file, return_path=True)
        weight_path = data_handler.read(weight_file, return_path=True)

        self.elmo = Elmo(option_path, weight_path, 1, requires_grad=trainable, dropout=dropout)

        self.project_dim = project_dim
        self.project_linear = None
        if project_dim:
            self.project_linear = nn.Linear(self.elmo.get_output_dim(), project_dim)

    @overrides
    def forward(self, chars):
        elmo_output = self.elmo(chars)
        elmo_representations = elmo_output["elmo_representations"][0]

        if self.project_linear:
            elmo_representations = self.project_linear(elmo_representations)
        return elmo_representations

    @overrides
    def get_output_dim(self):
        if self.project_linear:
            return self.project_dim
        return self.elmo.get_output_dim()
