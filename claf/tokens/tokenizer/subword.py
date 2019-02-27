
from pytorch_pretrained_bert.tokenization import WordpieceTokenizer, load_vocab

from claf.data.data_handler import CachePath, DataHandler

from .base import Tokenizer


class SubwordTokenizer(Tokenizer):
    """
    Subword Tokenizer

    text -> [word tokens] -> [[sub word tokens], ...]

    * Args:
        name: tokenizer name [wordpiece]
    """

    def __init__(self, name, word_tokenizer, config={}):
        super(SubwordTokenizer, self).__init__(name, f"subword-{name}+{word_tokenizer.cache_name}")
        self.data_handler = DataHandler(CachePath.VOCAB)
        self.config = config
        self.word_tokenizer = word_tokenizer
        self.subword_tokenizer = None

    """ Tokenizers """

    def _wordpiece(self, text, unit="text"):
        """
        ex) Hello World -> ['Hello', 'World'] -> ['He', '##llo', 'Wo', '##rld']
        """
        if self.subword_tokenizer is None:
            vocab_path = self.data_handler.read(self.config["vocab_path"], return_path=True)
            vocab = load_vocab(vocab_path)
            self.subword_tokenizer = WordpieceTokenizer(vocab)

        tokens = []

        if unit == "word":
            for sub_token in self.subword_tokenizer.tokenize(text):
                tokens.append(sub_token)
        else:
            for token in self.word_tokenizer.tokenize(text):
                for sub_token in self.subword_tokenizer.tokenize(token):
                    tokens.append(sub_token)

        return tokens
