
from claf.decorator import register
from claf.tokens import indexer, embedding
from claf.tokens.linguistic import POSTag, NER
from claf.tokens.token_maker import TokenMaker
from claf.tokens.tokenizer import PassText


def basic_embedding_fn(embedding_config, module):
    def wrapper(vocab):
        embedding_config["vocab"] = vocab
        return module(**embedding_config)

    return wrapper


@register(f"token:{TokenMaker.FEATURE_TYPE}")
class FeatureTokenMaker(TokenMaker):
    """
    Feature Token

    Do not use Embedding function.
    Just pass indexed_feature

    example.
        hello -> ['hello', 'world'] -> [3, 5] -> tensor

    consisting of
        - tokenizer: Tokenizer (need to define unit)
        - indexer: WordIndexer
        - embedding: None
        - vocab: Vocab
    """

    def __init__(self, tokenizers, indexer_config, embedding_config, vocab_config):
        tokenizer = PassText()
        do_tokenize = indexer_config.get("do_tokenize", False)
        if do_tokenize:
            text_unit = indexer_config.get("unit", None)
            if text_unit is None:
                raise ValueError("When use 'do_tokenize', 'unit' is required. ")

            del indexer_config["unit"]
            tokenizer = tokenizers[text_unit]

        super(FeatureTokenMaker, self).__init__(
            TokenMaker.FEATURE_TYPE,
            tokenizer=tokenizer,
            indexer=indexer.WordIndexer(tokenizer, **indexer_config),
            embedding_fn=None,
            vocab_config=vocab_config,
        )


@register(f"token:{TokenMaker.BERT_TYPE}")
class BertTokenMaker(TokenMaker):
    """
    BERT Token
    Pre-training of Deep Bidirectional Transformers for Language Understanding

    example.
        hello -> ['[CLS]', 'he', '##llo', [SEP]] -> [1, 4, 7, 2] -> BERT -> tensor

    consisting of
        - tokenizer: WordTokenizer
        - indexer: WordIndexer
        - embedding: ELMoEmbedding (Language Modeling BiLSTM)
        - vocab: Vocab
    """

    def __init__(self, tokenizers, indexer_config, embedding_config, vocab_config):
        tokenizer = tokenizers["subword"]
        super(BertTokenMaker, self).__init__(
            TokenMaker.BERT_TYPE,
            tokenizer=tokenizer,
            indexer=indexer.BertIndexer(tokenizer, **indexer_config),
            embedding_fn=basic_embedding_fn(embedding_config, embedding.BertEmbedding),
            vocab_config=vocab_config,
        )


@register(f"token:{TokenMaker.CHAR_TYPE}")
class CharTokenMaker(TokenMaker):
    """
    Character Token

    Character-level Convolutional Networks for Text Classification
    (https://arxiv.org/abs/1509.01626)

    example.
        hello -> ['h', 'e', 'l', 'l', 'o'] -> [2, 3, 4, 4, 5] -> CharCNN -> tensor

    consisting of
        - tokenizer: CharTokenizer
        - indexer: CharIndexer
        - embedding: CharEmbedding (CharCNN)
        - vocab: Vocab
    """

    def __init__(self, tokenizers, indexer_config, embedding_config, vocab_config):
        super(CharTokenMaker, self).__init__(
            TokenMaker.CHAR_TYPE,
            tokenizer=tokenizers["char"],
            indexer=indexer.CharIndexer(tokenizers["char"], **indexer_config),
            embedding_fn=basic_embedding_fn(embedding_config, embedding.CharEmbedding),
            vocab_config=vocab_config,
        )


@register(f"token:{TokenMaker.COVE_TYPE}")
class CoveTokenMaker(TokenMaker):
    """
    CoVe Token

    Learned in Translation: Contextualized Word Vectors (McCann et. al. 2017)
    (https://github.com/salesforce/cove)

    example.
        hello -> ['hello'] -> [2] -> CoVe -> tensor

    consisting of
        - tokenizer: WordTokenizer
        - indexer: WordIndexer
        - embedding: CoveEmbedding (Machine Translation LSTM)
        - vocab: Vocab
    """

    def __init__(self, tokenizers, indexer_config, embedding_config, vocab_config):
        super(CoveTokenMaker, self).__init__(
            TokenMaker.CHAR_TYPE,
            tokenizer=tokenizers["word"],
            indexer=indexer.WordIndexer(tokenizers["word"], **indexer_config),
            embedding_fn=basic_embedding_fn(embedding_config, embedding.CoveEmbedding),
            vocab_config=vocab_config,
        )


@register(f"token:{TokenMaker.ELMO_TYPE}")
class ElmoTokenMaker(TokenMaker):
    """
    ELMo Token
    Embedding from Language Modeling

    Deep contextualized word representations
    (https://github.com/allenai/allennlp/blob/master/allennlp/modules/elmo.py)

    example.
        hello -> ['h', 'e', 'l', 'l', 'o'] -> [2, 3, 4, 4, 5] -> ELMo -> tensor

    consisting of
        - tokenizer: WordTokenizer
        - indexer: WordIndexer
        - embedding: ELMoEmbedding (Language Modeling BiLSTM)
        - vocab: Vocab
    """

    def __init__(self, tokenizers, indexer_config, embedding_config, vocab_config):
        super(ElmoTokenMaker, self).__init__(
            TokenMaker.WORD_TYPE,
            tokenizer=tokenizers["word"],
            indexer=indexer.ELMoIndexer(tokenizers["word"], **indexer_config),
            embedding_fn=basic_embedding_fn(embedding_config, embedding.ELMoEmbedding),
            vocab_config="elmo",
        )


@register(f"token:{TokenMaker.EXACT_MATCH_TYPE}")
class ExactMatchTokenMaker(TokenMaker):
    """
    Exact Match Token (Sparse Feature)

    Three simple binary features, indicating whether p_i can be exactly matched
    to one question word in q, either in its original, lowercase or lemma form.

    example.
        c: i do, q: i -> ['i', 'do'] -> [1, 0] -> tensor

    consisting of
        - tokenizer: WordTokenizer
        - indexer: WordIndexer
        - embedding: SparseFeature
        - vocab: Vocab
    """

    def __init__(self, tokenizers, indexer_config, embedding_config, vocab_config):
        super(ExactMatchTokenMaker, self).__init__(
            TokenMaker.EXACT_MATCH_TYPE,
            tokenizer=tokenizers["word"],
            indexer=indexer.ExactMatchIndexer(tokenizers["word"], **indexer_config),
            embedding_fn=self._embedding_fn(embedding_config, indexer_config),
            vocab_config=vocab_config,
        )

    def _embedding_fn(self, embedding_config, indexer_config):
        def wrapper(vocab):
            embed_type = embedding_config.get("type", "sparse")
            if "type" in embedding_config:
                del embedding_config["type"]

            binary_classes = ["False", "True"]

            feature_count = 1  # origin
            embedding_config["classes"] = [binary_classes]

            if indexer_config.get("lower", False):
                feature_count += 1
                embedding_config["classes"].append(binary_classes)
            if indexer_config.get("lemma", False):
                feature_count += 1
                embedding_config["classes"].append(binary_classes)

            return embedding.SparseFeature(
                vocab, embed_type, feature_count, params=embedding_config
            )

        return wrapper


@register(f"token:{TokenMaker.WORD_TYPE}")
class WordTokenMaker(TokenMaker):
    """
    Word Token (default)

        i do -> ['i', 'do'] -> [1, 2] -> Embedding Matrix -> tensor

    consisting of
        - tokenizer: WordTokenizer
        - indexer: WordIndexer
        - embedding: WordEmbedding
        - vocab: Vocab
    """

    def __init__(self, tokenizers, indexer_config, embedding_config, vocab_config):
        super(WordTokenMaker, self).__init__(
            TokenMaker.WORD_TYPE,
            tokenizer=tokenizers["word"],
            indexer=indexer.WordIndexer(tokenizers["word"], **indexer_config),
            embedding_fn=basic_embedding_fn(embedding_config, embedding.WordEmbedding),
            vocab_config=vocab_config,
        )


@register(f"token:{TokenMaker.FREQUENT_WORD_TYPE}")
class FrequentWordTokenMaker(TokenMaker):
    """
    Frequent-Tuning Word Token

    word token + pre-trained word embeddings fixed and only fine-tune the N most frequent

    example.
        i do -> ['i', 'do'] -> [1, 2] -> Embedding Matrix -> tensor
        finetuning only 'do'

    consisting of
        - tokenizer: WordTokenizer
        - indexer: WordIndexer
        - embedding: FrequentTuningWordEmbedding
        - vocab: Vocab
    """

    def __init__(self, tokenizers, indexer_config, embedding_config, vocab_config):
        super(FrequentWordTokenMaker, self).__init__(
            TokenMaker.FREQUENT_WORD_TYPE,
            tokenizer=tokenizers["word"],
            indexer=indexer.WordIndexer(tokenizers["word"], **indexer_config),
            embedding_fn=basic_embedding_fn(
                embedding_config, embedding.FrequentTuningWordEmbedding
            ),
            vocab_config=vocab_config,
        )


@register(f"token:{TokenMaker.LINGUISTIC_TYPE}")
class LinguisticTokenMaker(TokenMaker):
    """
    Exact Match Token (Sparse Feature)

    Three simple binary features, indicating whether p_i can be exactly matched
    to one question word in q, either in its original, lowercase or lemma form.

    example.
        c: i do, q: i -> ['i', 'do'] -> [1, 0] -> tensor

    consisting of
        - tokenizer: WordTokenizer
        - indexer: WordIndexer
        - embedding: SparseFeature
        - vocab: Vocab
    """

    def __init__(self, tokenizers, indexer_config, embedding_config, vocab_config):
        super(LinguisticTokenMaker, self).__init__(
            TokenMaker.LINGUISTIC_TYPE,
            tokenizer=tokenizers["word"],
            indexer=indexer.LinguisticIndexer(tokenizers["word"], **indexer_config),
            embedding_fn=self._embedding_fn(embedding_config, indexer_config),
            vocab_config=vocab_config,
        )

    def _embedding_fn(self, embedding_config, indexer_config):
        def wrapper(vocab):
            embed_type = embedding_config.get("type", "sparse")
            if "type" in embedding_config:
                del embedding_config["type"]

            feature_count = 0
            embedding_config["classes"] = []

            if indexer_config.get("pos_tag", False):
                feature_count += 1
                embedding_config["classes"].append(POSTag.classes)
            if indexer_config.get("ner", False):
                feature_count += 1
                embedding_config["classes"].append(NER.classes)
            return embedding.SparseFeature(
                vocab, embed_type, feature_count, params=embedding_config
            )

        return wrapper
