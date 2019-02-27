
from pathlib import Path

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import MatrixSimilarity, SparseMatrixSimilarity

from tqdm import tqdm

from claf.decorator import register


@register("component:tfidf")
class TFIDF:
    """
    TF-IDF document retrieval model

    - Term Frequency
    - Inverse Document Frequency
    - log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))

    * Kwargs:
        k: the number of top k results
    """

    VOCAB_FNAME = "vocab.txt"
    TFIDF_FNAME = "tfidf.model"
    INDEX_FNAME = "similarities.index"

    def __init__(self, texts, word_tokenizer, k=1):
        super(TFIDF, self).__init__()
        self.k = k

        self.texts = texts
        self.word_tokenizer = word_tokenizer

    def init(self):
        corpus = [
            self.word_tokenizer.tokenize(text)
            for text in tqdm(self.texts, desc="make corpus (Tokenize)")
        ]
        self.vocab = Dictionary(corpus)
        self.init_model()

    def init_model(self):
        corpus = []
        for text in tqdm(self.texts, desc="make corpus (BoW)"):
            corpus.append(self.parse(text))

        self.model = TfidfModel(corpus)
        self.index = SparseMatrixSimilarity(self.model[corpus], num_features=len(self.vocab))

    def get_closest(self, query):
        query_tfidf = self.text_to_tfidf(query)

        self.index.num_best = self.k
        results = self.index[query_tfidf]

        return [
            (text_index, self.texts[text_index], score)  # return (index, text, score)
            for (text_index, score) in results
        ]

    def parse(self, query, ngram=1):
        query_tokens = self.word_tokenizer.tokenize(query)
        return self.vocab.doc2bow(query_tokens)

    def text_to_tfidf(self, query):
        """
        Create a tfidf-weighted word vector from query.

        tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
        """

        query_bow = self.parse(query)
        return self.model[query_bow]

    def save(self, dir_path):
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        vocab_path = str(dir_path / self.VOCAB_FNAME)
        model_path = str(dir_path / self.TFIDF_FNAME)
        index_path = str(dir_path / self.INDEX_FNAME)

        self.vocab.save(vocab_path)
        self.model.save(model_path)
        self.index.save(index_path)

    def load(self, dir_path):
        dir_path = Path(dir_path)

        vocab_path = str(dir_path / self.VOCAB_FNAME)
        model_path = str(dir_path / self.TFIDF_FNAME)
        index_path = str(dir_path / self.INDEX_FNAME)

        self.vocab = Dictionary.load(vocab_path)
        self.model = TfidfModel.load(model_path)
        self.index = SparseMatrixSimilarity.load(index_path)
