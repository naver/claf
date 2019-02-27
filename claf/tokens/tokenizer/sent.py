
import nltk.data

from .base import Tokenizer


class SentTokenizer(Tokenizer):
    """
    Sentence Tokenizer

    text -> [sent tokens]

    * Args:
        name: tokenizer name [punkt]
    """

    def __init__(self, name, config={}):
        super(SentTokenizer, self).__init__(name, f"sent-{name}")
        self.config = config

    """ Tokenizers """

    def _punkt(self, text, unit="text"):
        """
        ex) Hello World. This is punkt tokenizer -> ['Hello World', 'This is punkt tokenizer']
        """
        sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        return sent_tokenizer.tokenize(text)
