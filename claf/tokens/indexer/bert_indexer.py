
from overrides import overrides

from .base import TokenIndexer


class BertIndexer(TokenIndexer):
    """
    Bert Token Indexer

    * Property
        vocab: Vocab (claf.tokens.vocabulary)

    * Args:
        tokenizer: SubwordTokenizer

    * Kwargs:
        lowercase: word token to lowercase
        insert_start: insert start_token to first
        insert_end: append end_token
    """

    def __init__(self, tokenizer, do_tokenize=True):
        super(BertIndexer, self).__init__(tokenizer)
        self.do_tokenize = do_tokenize

    @overrides
    def index(self, text):
        input_type = type(text)
        if input_type == str:
            return self._index_text(text)
        elif input_type == list:
            texts = text  # List of text case
            return [self._index_text(text) for text in texts]
        else:
            raise ValueError(f"Not supported type: {type(text)}")

    def _index_text(self, text):
        if self.do_tokenize:
            tokens = self.tokenizer.tokenize(text)
        else:
            tokens = [text]

        indexed_tokens = [self.vocab.get_index(token) for token in tokens]

        # Insert CLS_TOKEN ans SEP_TOKEN
        insert_start = self.vocab.get_index(self.vocab.cls_token)
        indexed_tokens.insert(0, insert_start)

        insert_end = self.vocab.get_index(self.vocab.sep_token)
        indexed_tokens.append(insert_end)
        return indexed_tokens
