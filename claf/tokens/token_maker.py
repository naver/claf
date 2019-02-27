class TokenMaker:
    """
    Token Maker (Data Transfer Object)

    Token Maker consists of Tokenizer, Indexer, Embedding and Vocab

    * Kwargs:
        tokenizer: Tokenizer (claf.tokens.tokenizer.base)
        indexer: TokenIndexer (claf.tokens.indexer.base)
        embedding_fn: wrapper function of TokenEmbedding (claf.tokens.embedding.base)
        vocab_config: config dict of Vocab (claf.tokens.vocaburary)
    """

    # Token Type List
    FEATURE_TYPE = "feature"  # Do not use embedding, pass indexed_feature
    BERT_TYPE = "bert"
    CHAR_TYPE = "char"
    COVE_TYPE = "cove"
    ELMO_TYPE = "elmo"
    EXACT_MATCH_TYPE = "exact_match"
    WORD_TYPE = "word"
    FREQUENT_WORD_TYPE = "frequent_word"
    LINGUISTIC_TYPE = "linguistic"

    def __init__(
        self, token_type, tokenizer=None, indexer=None, embedding_fn=None, vocab_config=None
    ):
        self.type_name = token_type
        self._tokenizer = tokenizer
        self._indexer = indexer
        self._embedding_fn = embedding_fn
        self._vocab_config = vocab_config

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer):
        self._tokenizer = tokenizer

    @property
    def indexer(self):
        return self._indexer

    @indexer.setter
    def indexer(self, indexer):
        self._indexer = indexer

    @property
    def embedding_fn(self):
        return self._embedding_fn

    @embedding_fn.setter
    def embedding_fn(self, embedding_fn):
        self._embedding_fn = embedding_fn

    @property
    def vocab_config(self):
        return self._vocab_config

    @vocab_config.setter
    def vocab_config(self, vocab_config):
        self._vocab_config = vocab_config

    @property
    def vocab(self):
        return self._vocab

    @vocab.setter
    def vocab(self, vocab):
        self._vocab = vocab

    def set_vocab(self, vocab):
        self._indexer.set_vocab(vocab)
        self._vocab = vocab
