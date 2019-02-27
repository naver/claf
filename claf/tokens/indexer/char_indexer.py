
from overrides import overrides

from .base import TokenIndexer


class CharIndexer(TokenIndexer):
    """
    Character Token Indexer

    * Property
        vocab: Vocab (claf.tokens.vocabulary)

    * Args:
        tokenizer: CharTokenizer

    * Kwargs:
        insert_char_start: insert start index (eg. ['h', 'i'] -> ['<s>', 'h', 'i'] )
            default is None
        insert_char_end: insert end index (eg. ['h', 'i'] -> ['h', 'i', '</s>'] )
            default is None
    """

    def __init__(self, tokenizer, insert_char_start=None, insert_char_end=None):
        super(CharIndexer, self).__init__(tokenizer)

        self.insert_char_start = insert_char_start
        self.insert_char_end = insert_char_end

    @overrides
    def index(self, text):
        indexed_tokens = [self.index_token(token) for token in self.tokenizer.tokenize(text)]
        return indexed_tokens

    def index_token(self, chars):
        char_ids = [self.vocab.get_index(char) for char in chars]

        if self.insert_char_start is not None:
            char_ids.insert(0, self.vocab.get_index(self.vocab.start_token))
        if self.insert_char_end is not None:
            char_ids.append(self.vocab.get_index(self.vocab.end_token))
        return char_ids
