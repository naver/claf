
from collections import defaultdict

from claf.data.data_handler import CachePath, DataHandler


class VocabDict(defaultdict):
    """
    Vocab DefaultDict Class

    * Kwargs:
        oov_value: out-of-vocaburary token value (eg. <unk>)
    """

    def __init__(self, oov_value):
        self.oov_value = oov_value

    def __missing__(self, key):
        return self.oov_value


class Vocab:
    """
    Vocaburary Class

    Vocab consists of token_to_index and index_to_token.

    * Args:
        token_name: Token name (Token and Vocab is one-to-one relationship)

    * Kwargs:
        pad_token: padding token value (eg. <pad>)
        oov_token: out-of-vocaburary token value (eg. <unk>)
        start_token: start token value (eg. <s>, <bos>)
        end_token: end token value (eg. </s>, <eos>)
        cls_token: CLS token value for BERT (eg. [CLS])
        sep_token: SEP token value for BERT (eg. [SEP])
        min_count: token's minimal frequent count.
            when you define min_count, tokens remain that bigger than min_count.
        max_vocab_size: vocaburary's maximun size.
            when you define max_vocab_size, tokens are selected according to frequent count.
        frequent_count: get frequent_count threshold_index.
            (eg. frequent_count = 1000, threshold_index is the tokens that frequent_count is 999 index number.)
        pretrained_path: pretrained vocab file path
            (format: A\nB\nC\nD\n...)
    """

    DEFAULT_PAD_INDEX, DEFAULT_PAD_TOKEN = 0, "[PAD]"
    DEFAULT_OOV_INDEX, DEFAULT_OOV_TOKEN = 1, "[UNK]"

    # pretrained_vocab handle methods
    PRETRAINED_ALL = "all"  # Case. embedding matrix - predefine_vocab fixed
    PRETRAINED_INTERSECT = "intersect"  # add token that included in predefine_vocab, else UNK_token

    def __init__(
        self,
        token_name,
        pad_token=None,
        oov_token=None,
        start_token=None,
        end_token=None,
        cls_token=None,
        sep_token=None,
        min_count=None,
        max_vocab_size=None,
        frequent_count=None,
        pretrained_path=None,
        pretrained_token=None,
    ):
        self.token_name = token_name

        # basic token (pad and oov)
        self.pad_index = self.DEFAULT_PAD_INDEX
        self.pad_token = pad_token
        if pad_token is None:
            self.pad_token = self.DEFAULT_PAD_TOKEN

        self.oov_index = self.DEFAULT_OOV_INDEX
        self.oov_token = oov_token
        if oov_token is None:
            self.oov_token = self.DEFAULT_OOV_TOKEN

        # special_tokens
        self.start_token = start_token
        self.end_token = end_token
        self.cls_token = cls_token
        self.sep_token = sep_token

        self.min_count = min_count
        self.max_vocab_size = max_vocab_size

        self.token_counter = None
        self.frequent_count = frequent_count
        self.threshold_index = None

        self.pretrained_path = pretrained_path
        self.pretrained_token = pretrained_token
        self.pretrained_token_methods = [self.PRETRAINED_ALL, self.PRETRAINED_INTERSECT]

    def init(self):
        self.token_to_index = VocabDict(self.oov_index)
        self.index_to_token = VocabDict(self.oov_token)

        # add default token (pad, oov)
        self.add(self.pad_token)
        self.add(self.oov_token)

        special_tokens = [self.start_token, self.end_token, self.cls_token, self.sep_token]
        for token in special_tokens:
            if token is not None:
                self.add(token)

    def build(self, token_counter, predefine_vocab=None):
        """
        build token with token_counter

        * Args:
            token_counter: (collections.Counter) token's frequent_count Counter.
        """

        if predefine_vocab is not None:
            if (
                self.pretrained_token is None
                or self.pretrained_token not in self.pretrained_token_methods
            ):
                raise ValueError(
                    f"When use 'predefine_vocab', need to set 'pretrained_token' {self.pretrained_token_methods}"
                )

        if predefine_vocab:
            if self.pretrained_token == self.PRETRAINED_ALL:
                self.from_texts(predefine_vocab)
                return
            else:
                predefine_vocab = set(predefine_vocab)

        self.token_counter = token_counter
        self.init()

        token_counts = list(token_counter.items())
        token_counts.sort(key=lambda x: x[1], reverse=True)  # order: DESC

        if self.max_vocab_size is not None:
            token_counts = token_counts[: self.max_vocab_size]

        for token, count in token_counts:
            if self.min_count is not None:
                if count >= self.min_count:
                    self.add(token, predefine_vocab=predefine_vocab)
            else:
                self.add(token, predefine_vocab=predefine_vocab)

            if self.threshold_index is None and self.frequent_count is not None:
                if count < self.frequent_count:
                    self.threshold_index = len(self.token_to_index)

    def build_with_pretrained_file(self, token_counter):
        data_handler = DataHandler(CachePath.VOCAB)
        vocab_texts = data_handler.read(self.pretrained_path)
        predefine_vocab = vocab_texts.split("\n")

        self.build(token_counter, predefine_vocab=predefine_vocab)

    def __len__(self):
        return len(self.token_to_index)

    def add(self, token, predefine_vocab=None):
        if token in self.token_to_index:
            return  # already added
        if predefine_vocab:
            if self.pretrained_token == self.PRETRAINED_INTERSECT and token not in predefine_vocab:
                return

        index = len(self.token_to_index)

        self.token_to_index[token] = index
        self.index_to_token[index] = token

    def get_index(self, token):
        return self.token_to_index[token]

    def get_token(self, index):
        return self.index_to_token[index]

    def get_all_tokens(self):
        return list(self.token_to_index.keys())

    def dump(self, path):
        with open(path, "w", encoding="utf-8") as out_file:
            out_file.write(self.to_text())

    def load(self, path):
        with open(path, "r", encoding="utf-8") as in_file:
            texts = in_file.read()

        self.from_texts(texts)

    def to_text(self):
        return "\n".join(self.get_all_tokens())

    def from_texts(self, texts):
        if type(texts) == list:
            tokens = texts
        else:
            tokens = [token for token in texts.split("\n")]
        tokens = [token for token in tokens if token]  # filtering empty string

        # basic token (pad and oov)
        if self.pad_token in tokens:
            self.pad_index = tokens.index(self.pad_token)
        else:
            self.pad_index = len(tokens)
            tokens.append(self.pad_token)

        if self.oov_token in tokens:
            self.oov_index = tokens.index(self.oov_token)
        else:
            self.oov_index = len(tokens)
            tokens.append(self.oov_token)

        self.token_to_index = VocabDict(self.oov_index)
        self.index_to_token = VocabDict(self.oov_token)

        for token in tokens:
            self.add(token)
        return self
