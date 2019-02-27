

import re

from overrides import overrides

from claf import utils as common_utils

from .base import Tokenizer


class WordTokenizer(Tokenizer):
    """
    Word Tokenizer

    * Args:
        name: tokenizer name [treebank_en|spacy_en|mecab_ko|bert_basic]

    * Kwargs:
        flatten: return type as flatten list
        split_with_regex: post split action. Split tokens that the tokenizer cannot split.
    """

    def __init__(self, name, sent_tokenizer, config={}, split_with_regex=True):
        super(WordTokenizer, self).__init__(name, f"word-{name}+{sent_tokenizer.cache_name}")
        self.config = config
        self.sent_tokenizer = sent_tokenizer
        self.word_tokenizer = None

        self.split_with_regex = split_with_regex
        if split_with_regex:
            self.extra_split_chars_re = self.make_split_regex_expression()

    def make_split_regex_expression(self):
        """
        Apply a small amount of extra splitting to the given tokens, this is in particular to avoid UNK tokens
        due to contraction, quotation, or other forms of puncutation. I haven't really done tests to see
        if/how much difference this makes, but it does avoid some common UNKs I noticed in SQuAD/TriviaQA
        """
        extra_split_chars = (
            "-",
            "£",
            "€",
            "¥",
            "¢",
            "₹",
            "*",
            "\u2212",
            "\u2014",
            "\u2013",
            "/",
            "~",
            '"',
            "'",
            "\ud01C",
            "\u2019",
            "\u201D",
            "\u2018",
            "\u00B0",
            ".",
            ":",
        )
        extra_split_tokens = (
            "``",
            "(?<=[^_])_(?=[^_])",  # dashes w/o a preceeding or following dash, so __wow___ -> ___ wow ___
            "''",
            "[" + "".join(extra_split_chars) + "]",
        )
        return re.compile("(" + "|".join(extra_split_tokens) + ")")

    @overrides
    def _tokenize(self, text, unit="text"):
        """ Text -> word tokens """
        if type(text) != str:
            raise ValueError(f"text type is must be str. not {type(text)}")

        if unit == "sentence":
            tokens = getattr(self, f"_{self.name}")(text)
        else:
            sentences = self.sent_tokenizer.tokenize(text)
            tokens = [getattr(self, f"_{self.name}")(sentence) for sentence in sentences]

        if self.split_with_regex and self.name != "spacy_en":
            tokens = self._split_with_regex(tokens)

        return list(common_utils.flatten(tokens))

    def _split_with_regex(self, sentences):
        for i, sentence in enumerate(sentences):
            sentences[i] = [token for token in self._post_split_tokens(sentence)]
        return sentences

    def _post_split_tokens(self, tokens):
        return [[x for x in self.extra_split_chars_re.split(token) if x != ""] for token in tokens]

    """ Tokenizers """

    def _space_all(self, text):
        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        prev_is_whitespace = True
        tokens = []
        for char in text:
            if is_whitespace(char):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    tokens.append(char)
                else:
                    tokens[-1] += char
                prev_is_whitespace = False
        return tokens

    def _treebank_en(self, text):
        if self.word_tokenizer is None:
            import nltk

            self.word_tokenizer = nltk.TreebankWordTokenizer()

        return [
            token.replace("''", '"').replace("``", '"')
            for token in self.word_tokenizer.tokenize(text)
        ]

    def _spacy_en(self, text):
        if self.word_tokenizer is None:
            from claf.tokens.tokenizer.utils import load_spacy_model_for_tokenizer

            self.word_tokenizer = load_spacy_model_for_tokenizer(self.extra_split_chars_re)

        def _remove_spaces(tokens):
            return [token.text for token in tokens if not token.is_space]

        return _remove_spaces(self.word_tokenizer(text))

    def _bert_basic(self, text):
        if self.word_tokenizer is None:
            from pytorch_pretrained_bert.tokenization import BasicTokenizer

            self.word_tokenizer = BasicTokenizer(**self.config)

        return self.word_tokenizer.tokenize(text)

    def _mecab_ko(self, text):
        if self.word_tokenizer is None:
            from konlpy.tag import Mecab

            self.word_tokenizer = Mecab()

        return self.word_tokenizer.morphs(text)
