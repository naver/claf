class TokenIndexer:
    """
    Token Indexer

    indexing tokens (eg. 'hi' -> 4)
    """

    def __init__(self, tokenizer):
        self.param_key = None
        self.tokenizer = tokenizer

    def index(self, token):
        """ indexing function """
        raise NotImplementedError

    def set_vocab(self, vocab):
        self.vocab = vocab
