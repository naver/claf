class Tokenizer:
    """
    Tokenizer Base Class
    """

    MAX_TO_KEEP_CACHE = 3

    def __init__(self, name, cache_name):
        self.cache = {}  # dict: {text: tokenized_tokens}
        self.name = name
        self.cache_name = cache_name

    def tokenize(self, text, unit="text"):
        if type(text) == str and text in self.cache:
            return self.cache[text]

        tokenized_tokens = self._tokenize(text, unit="text")

        # Cache
        if len(self.cache) <= self.MAX_TO_KEEP_CACHE:
            self.cache[text] = tokenized_tokens
        else:
            first_key = next(iter(self.cache.keys()))
            del self.cache[first_key]

        return tokenized_tokens

    def _tokenize(self, text, unit="text"):
        """ splitting text into tokens. """
        if type(text) != str:
            raise ValueError(f"text type is must be str. not {type(text)}")

        return getattr(self, f"_{self.name}")(text, unit=unit)
