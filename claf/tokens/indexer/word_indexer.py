
from overrides import overrides

from .base import TokenIndexer


class WordIndexer(TokenIndexer):
    """
    Word Token Indexer

    * Property
        vocab: Vocab (claf.tokens.vocabulary)

    * Args:
        tokenizer: WordTokenizer

    * Kwargs:
        lowercase: word token to lowercase
        insert_start: insert start_token to first
        insert_end: append end_token
    """

    def __init__(
        self, tokenizer, do_tokenize=True, lowercase=False, insert_start=None, insert_end=None
    ):
        super(WordIndexer, self).__init__(tokenizer)

        self.do_tokenize = do_tokenize
        self.lowercase = lowercase

        self.insert_start = insert_start
        self.insert_end = insert_end

    @overrides
    def index(self, text):
        input_type = type(text)
        if input_type == str:
            indexed_tokens = self._index_text(text)
        elif input_type == list:
            indexed_tokens = self._index_list_of_text(text)
        else:
            raise ValueError(f"Not supported type: {type(text)}")

        if self.insert_start is not None:
            insert_start = self.vocab.get_index(self.vocab.start_token)
            indexed_tokens.insert(0, insert_start)
        if self.insert_end is not None:
            insert_end = self.vocab.get_index(self.vocab.end_token)
            indexed_tokens.append(insert_end)
        return indexed_tokens

    def _index_text(self, text):
        if not self.do_tokenize:
            raise ValueError("input text type is 'str'. 'do_tokenize' is required.")

        return [self._index_token(token) for token in self.tokenizer.tokenize(text)]

    def _index_list_of_text(self, list_of_text):
        if self.do_tokenize:
            indexed_tokens = [
                [self._index_token(token) for token in self.tokenizer.tokenize(text)]
                for text in list_of_text
            ]
        else:
            indexed_tokens = [self._index_token(text) for text in list_of_text]
        return indexed_tokens

    def _index_token(self, token):
        if self.lowercase:
            token = token.lower()

        return self.vocab.get_index(token)
