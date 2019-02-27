
from claf.tokens import hangul as hg

from .base import Tokenizer


class CharTokenizer(Tokenizer):
    """
    Character Tokenizer

    text -> word tokens -> [char tokens]

    * Args:
        name: tokenizer name [character|decompose_ko]
        word_tokenizer: word tokenizer object
    """

    def __init__(self, name, word_tokenizer, config={}):
        super(CharTokenizer, self).__init__(name, f"char-{name}+{word_tokenizer.cache_name}")
        self.config = config
        self.word_tokenizer = word_tokenizer

    """ Tokenizers """

    def _character(self, text, unit="text"):
        """
        ex) Hello World -> ['Hello', 'World'] -> [['H', 'e', 'l', 'l', 'o'], ['W', 'o', 'r', 'l', 'd']]
        """
        if unit == "word":
            return [char for char in text]
        else:
            return [[char for char in word] for word in self.word_tokenizer.tokenize(text)]

    def _jamo_ko(self, text, unit="text"):
        """
        ex) 안녕 세상 -> ['안녕', '세상'] -> [['ㅇ', 'ㅏ', 'ㄴ', 'ㄴ', 'ㅕ', 'ㅇ'], ['ㅅ', 'ㅔ', 'ㅅ', 'ㅏ', 'ㅇ']]
        """

        def decompose(char):
            if hg.is_hangul(char):
                try:
                    return [c for c in hg.decompose(char) if c != ""]
                except IndexError:  # Case: ㅋㅋㅋㅋ
                    return [char]
            else:
                return [char]

        tokens = []
        if unit == "word":
            chars = []
            for char in text:
                chars.extend(decompose(char))
            tokens.append(chars)
        else:
            for word in self.word_tokenizer.tokenize(text):
                chars = []
                for char in word:
                    chars.extend(decompose(char))
                tokens.append(chars)
        return tokens
