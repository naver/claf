
from collections import Counter
import logging
import time

from tqdm import tqdm

from claf.data.data_handler import CachePath, DataHandler
from claf.data.utils import padding_tokens, transpose
from claf.tokens.token_maker import TokenMaker
from claf.tokens.vocabulary import Vocab
from claf import utils as common_utils

logger = logging.getLogger(__name__)


class TextHandler:
    """
    Text Handler

    - voacb and token_counter
    - raw_features -> indexed_features
    - raw_features -> tensor

    * Args:
        token_makers: Dictionary consisting of
            - key: token_name
            - value: TokenMaker (claf.tokens.token_maker)

    * Kwargs:
        lazy_indexing: Apply `Lazy Evaluation` to text indexing
    """

    def __init__(self, token_makers, lazy_indexing=True):
        self.token_makers = token_makers
        self.lazy_indexing = lazy_indexing

        self.data_handler = DataHandler(cache_path=CachePath.TOKEN_COUNTER)

    def build_vocabs(self, token_counters):
        logger.info("Start build vocab")
        vocab_start_time = time.time()

        vocabs = {}
        for token_name, token_maker in self.token_makers.items():
            is_defined_config = type(token_maker.vocab_config) == dict
            if is_defined_config:
                token_counter = token_counters[token_name]
                vocab = self._build_vocab_with_config(token_name, token_maker, token_counter)
            else:
                vocab = Vocab(token_name)
                vocab.init()

            vocabs[token_name] = vocab
            logger.info(
                f" => {token_name} vocab size: {len(vocab)}  (use predefine vocab: {vocab.pretrained_path is not None})"
            )

        vocab_elapased_time = time.time() - vocab_start_time
        logger.info(f"Complete build vocab...  elapsed_time: {vocab_elapased_time}\n")

        # Setting Indexer (vocab)
        for token_name, token_maker in self.token_makers.items():
            token_maker.set_vocab(vocabs[token_name])
        return vocabs

    def _build_vocab_with_config(self, token_name, token_maker, token_counter):
        token_maker.vocab_config["token_name"] = token_name
        vocab = Vocab(**token_maker.vocab_config)

        if vocab.pretrained_path is not None:
            vocab.build_with_pretrained_file(token_counter)
        else:
            vocab.build(token_counter)
        return vocab

    def make_token_counters(self, texts, config=None):
        token_counters = {}
        for token_name, token_maker in self.token_makers.items():
            token_vocab_config = token_maker.vocab_config
            if type(token_vocab_config) == dict:
                if token_vocab_config.get("pretrained_token", None) == Vocab.PRETRAINED_ALL:
                    texts = [
                        ""
                    ]  # do not use token_counter from dataset -> make empty token_counter

            token_counter = self._make_token_counter(
                texts, token_maker.tokenizer, config=config, desc=f"{token_name}-vocab"
            )
            logger.info(f" * {token_name} token_counter size: {len(token_counter)}")

            token_counters[token_name] = token_counter
        return token_counters

    def _make_token_counter(self, texts, tokenizer, config=None, desc=None):
        tokenizer_name = tokenizer.name

        cache_token_counter = None
        if config is not None:
            data_reader_config = config.data_reader
            cache_token_counter = self.data_handler.cache_token_counter(
                data_reader_config, tokenizer_name
            )

        if cache_token_counter:
            return cache_token_counter
        else:
            tokens = [
                token for text in tqdm(texts, desc=desc) for token in tokenizer.tokenize(text)
            ]
            flatten_list = list(common_utils.flatten(tokens))
            token_counter = Counter(flatten_list)

            if config is not None:  # Cache TokenCounter
                self.data_handler.cache_token_counter(
                    data_reader_config, tokenizer_name, obj=token_counter
                )
            return token_counter

    def index(self, datas, text_columns):
        logger.info(f"Start token indexing, Lazy: {self.lazy_indexing}")
        indexing_start_time = time.time()

        for data_type, data in datas.items():
            self._index_features(
                data.features, text_columns, desc=f"indexing features ({data_type})"
            )

        indexing_elapased_time = time.time() - indexing_start_time
        logger.info(f"Complete token indexing... elapsed_time: {indexing_elapased_time} \n")

    def _index_features(self, features, text_columns, desc=None, suppress_tqdm=False):
        for feature in tqdm(features, desc=desc, disable=suppress_tqdm):
            for key, text in feature.items():
                if key not in text_columns:
                    continue

                # Set data_type (text => {"text": ..., "token1": ..., ...})
                if type(feature[key]) != dict:
                    feature[key] = {"text": text}
                if type(text) == dict:
                    text = text["text"]

                for token_name, token_maker in self.token_makers.items():
                    param_key = token_maker.indexer.param_key
                    if param_key == key:
                        continue

                    feature[key][token_name] = self._index_token(token_maker, text, feature)

    def _index_token(self, token_maker, text, data):
        def index():
            indexer = token_maker.indexer
            params = {}
            if token_maker.type_name == TokenMaker.EXACT_MATCH_TYPE:
                param_text = data[indexer.param_key]
                if type(param_text) == dict:
                    param_text = param_text["text"]
                params["query_text"] = param_text
            return indexer.index(text, **params)

        if self.lazy_indexing:
            return index
        else:
            return index()

    def raw_to_tensor_fn(self, data_reader, cuda_device=None, helper={}):
        def raw_to_tensor(inputs):
            is_one = True  # batch_size 1 flag
            feature, _helper = data_reader.read_one_example(inputs)

            nonlocal helper
            helper.update(_helper)

            if type(feature) == list:
                is_one = False
                features = feature
            else:
                features = [feature]

            self._index_features(features, data_reader.text_columns, suppress_tqdm=True)

            if is_one:
                indexed_features = features[0]
            else:  # when features > 1, need to transpose (dict_of_list -> list_of_dict)
                indexed_features = {}
                for key in features[0]:
                    feature_with_key = [feature[key] for feature in features]
                    indexed_features[key] = transpose(feature_with_key, skip_keys=["text"])

            for key in indexed_features:
                for token_name in self.token_makers:
                    if token_name not in indexed_features[key]:
                        continue

                    indexed_values = indexed_features[key][token_name]
                    if is_one:
                        indexed_values = [indexed_values]

                    tensor = padding_tokens(indexed_values, token_name=token_name)
                    if cuda_device is not None and type(tensor) != list:
                        tensor = tensor.cuda(cuda_device)
                    indexed_features[key][token_name] = tensor

            for key in indexed_features:
                if "text" in indexed_features[key]:
                    del indexed_features[key]["text"]

            return indexed_features, helper

        return raw_to_tensor
