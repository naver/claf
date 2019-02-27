"""
This code is from allenai/allennlp
(https://github.com/allenai/allennlp/blob/master/allennlp/data/token_indexers/elmo_indexer.py)
"""

from overrides import overrides

from .base import TokenIndexer


def _make_bos_eos(
    character: int,
    padding_character: int,
    beginning_of_word_character: int,
    end_of_word_character: int,
    max_word_length: int,
):
    char_ids = [padding_character] * max_word_length
    char_ids[0] = beginning_of_word_character
    char_ids[1] = character
    char_ids[2] = end_of_word_character
    return char_ids


class ELMoIndexer(TokenIndexer):
    """
    Maps individual tokens to sequences of character ids, compatible with ELMo.
    To be consistent with previously trained models, we include it here as special of existing
    character indexers.
    """

    max_word_length = 50

    # char ids 0-255 come from utf-8 encoding bytes
    # assign 256-300 to special chars
    beginning_of_sentence_character = 256  # <begin sentence>
    end_of_sentence_character = 257  # <end sentence>
    beginning_of_word_character = 258  # <begin word>
    end_of_word_character = 259  # <end word>
    padding_character = 260  # <padding><Paste>

    beginning_of_sentence_characters = _make_bos_eos(
        beginning_of_sentence_character,
        padding_character,
        beginning_of_word_character,
        end_of_word_character,
        max_word_length,
    )

    end_of_sentence_characters = _make_bos_eos(
        end_of_sentence_character,
        padding_character,
        beginning_of_word_character,
        end_of_word_character,
        max_word_length,
    )

    BOS_TOKEN = "<S>"
    EOS_TOKEN = "</S>"

    def __init__(self, tokenizer):
        super(ELMoIndexer, self).__init__(tokenizer)

    @overrides
    def index(self, text):
        indexed_tokens = [self.index_token(token) for token in self.tokenizer.tokenize(text)]
        return indexed_tokens

    def index_token(self, word):
        if word == self.BOS_TOKEN:
            char_ids = self.beginning_of_sentence_characters
        elif word == self.EOS_TOKEN:
            char_ids = self.end_of_sentence_characters
        else:
            word_encodeds = word.encode("utf-8", "ignore")[: (self.max_word_length - 2)]
            char_ids = [char_id for char_id in word_encodeds]
            char_ids = [self.beginning_of_word_character] + char_ids + [self.end_of_word_character]
        return [c + 1 for c in char_ids]
