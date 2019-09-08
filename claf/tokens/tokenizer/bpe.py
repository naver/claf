
from pytorch_transformers import RobertaTokenizer

from claf.data.data_handler import CachePath, DataHandler

from .base import Tokenizer


class BPETokenizer(Tokenizer):
    """
    BPTE(Byte-Pair Encoding) Tokenizer
    text -> ...
    * Args:
        name: tokenizer name [roberta]
    """

    def __init__(self, name, config={}):
        super(BPETokenizer, self).__init__(name, f"bpe-{name}")
        self.data_handler = DataHandler(CachePath.VOCAB)
        self.config = config

        self.bpe_tokenizer = None

    """ Tokenizers """

    def _roberta(self, text, unit="text"):
        """
        ex)
        """
        if self.bpe_tokenizer is None:
            vocab_path = self.data_handler.read(self.config["vocab_path"], return_path=True)
            merges_path = self.data_handler.read(self.config["merges_path"], return_path=True)
            del self.config["vocab_path"]
            del self.config["merges_path"]

            self.bpe_tokenizer = RobertaTokenizer(vocab_path, merges_path, **self.config)

        return self.bpe_tokenizer._tokenize(text)

