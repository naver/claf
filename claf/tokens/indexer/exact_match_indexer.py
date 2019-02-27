

from overrides import overrides
from nltk.stem import WordNetLemmatizer

from .base import TokenIndexer


class ExactMatchIndexer(TokenIndexer):
    """
    Exact Match Token Indexer

    * Property
        vocab: Vocab (claf.tokens.vocabulary)

    * Args:
        tokenizer: WordTokenizer

    * Kwargs:
        lower: add lower feature. default is True (0 or 1)
        lemma: add lemma case feature. feature is True (0 or 1)
    """

    def __init__(self, tokenizer, lower=True, lemma=True):
        super(ExactMatchIndexer, self).__init__(tokenizer)

        self.param_key = "question"
        self.lemmatizer = WordNetLemmatizer()

        self.lower = lower
        self.lemma = lemma

    @overrides
    def index(self, text, query_text):
        tokenized_query_text = self.tokenizer.tokenize(query_text)
        query_tokens = {
            "origin": set(tokenized_query_text),
            "lower": set([token.lower() for token in tokenized_query_text]),
            "lemma": set(
                [self.lemmatizer.lemmatize(token.lower()) for token in tokenized_query_text]
            ),
        }

        indexed_tokens = [
            self.index_token(token, query_tokens) for token in self.tokenizer.tokenize(text)
        ]
        return indexed_tokens

    def index_token(self, token, query_tokens):
        em_feature = []

        # 1. origin
        origin_case = 1 if token in query_tokens["origin"] else 0
        em_feature.append(origin_case + 2)

        # 2. lower
        if self.lower:
            lower_case = 1 if token.lower() in query_tokens["lower"] else 0
            em_feature.append(lower_case + 2)

        # 3. lemma
        if self.lemma:
            lemma_case = (
                1 if self.lemmatizer.lemmatize(token.lower()) in query_tokens["lemma"] else 0
            )
            em_feature.append(lemma_case + 2)
        return em_feature
