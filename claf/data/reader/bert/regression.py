
import logging
import uuid

from overrides import overrides
from tqdm import tqdm

from claf.data.batch import make_batch
from claf.data.dataset import RegressionBertDataset
from claf.data.reader.base import DataReader
from claf.data import utils
from claf.decorator import register

logger = logging.getLogger(__name__)


@register("reader:regression_bert")
class RegressionBertReader(DataReader):
    """
    Regression DataReader for BERT

    * Args:
        file_paths: .tsv file paths (train and dev)
        tokenizers: defined tokenizers config
    """

    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    UNK_TOKEN = "[UNK]"

    def __init__(
        self,
        file_paths,
        tokenizers,
        sequence_max_length=None,
        label_key="score",
        is_test=False,
    ):

        super(RegressionBertReader, self).__init__(file_paths, RegressionBertDataset)

        self.sequence_max_length = sequence_max_length
        self.text_columns = ["bert_input", "sequence"]

        if "subword" not in tokenizers:
            raise ValueError("WordTokenizer and SubwordTokenizer is required.")

        self.subword_tokenizer = tokenizers["subword"]
        self.label_key = label_key
        self.is_test = is_test

    def _get_data(self, file_path, **kwargs):
        data = self.data_handler.read(file_path)
        seq_cls_data = json.loads(data)

        return seq_cls_data["data"]

    @overrides
    def _read(self, file_path, data_type=None):
        """
        .json file structure should be something like this:

        {
            "data": [
                {
                    "sequence_a": "what a wonderful day!",
                    "sequence_b": "what a great day!",
                    "score": 0.9
                },
                ...
            ]
        }
        """

        data = self._get_data(file_path, data_type=data_type)

        helper = {
            "file_path": file_path,
            "examples": {},
            "cls_token": self.CLS_TOKEN,
            "sep_token": self.SEP_TOKEN,
            "unk_token": self.UNK_TOKEN,
            "model": {

            },
            "predict_helper": {
            }
        }
        features, labels = [], []

        for example in tqdm(data, desc=data_type):
            sequence_a = utils.get_sequence_a(example)
            sequence_b = example.get("sequence_b", None)

            sequence_a_sub_tokens = self.subword_tokenizer.tokenize(sequence_a)
            sequence_b_sub_tokens = None
            bert_input = [self.CLS_TOKEN] + sequence_a_sub_tokens + [self.SEP_TOKEN]

            if sequence_b is not None:
                sequence_b_sub_tokens = self.subword_tokenizer.tokenize(sequence_b)
                bert_input += sequence_b_sub_tokens + [self.SEP_TOKEN]

            if (
                    self.sequence_max_length is not None
                    and data_type == "train"
                    and len(bert_input) > self.sequence_max_length
            ):
                continue

            if "uid" in example:
                data_uid = example["uid"]
            else:
                data_uid = str(uuid.uuid1())

            feature_row = {
                "id": data_uid,
                "bert_input": bert_input,
            }
            features.append(feature_row)

            score = example[self.label_key]
            label_row = {
                "id": data_uid,
                "score": score,
            }
            labels.append(label_row)

            helper["examples"][data_uid] = {
                "sequence_a": sequence_a,
                "sequence_a_sub_tokens": sequence_a_sub_tokens,
                "sequence_b": sequence_b,
                "sequence_b_sub_tokens": sequence_b_sub_tokens,
                "score": score,
            }

            if self.is_test and len(features) >= 10:
                break

        return make_batch(features, labels), helper

    def read_one_example(self, inputs):
        """ inputs keys: sequence_a and sequence_b """
        sequence_a = utils.get_sequence_a(inputs)
        sequence_b = inputs.get("sequence_b", None)

        sequence_a_sub_tokens = self.subword_tokenizer.tokenize(sequence_a)
        bert_input = [self.CLS_TOKEN] + sequence_a_sub_tokens + [self.SEP_TOKEN]

        if sequence_b:
            sequence_b_sub_tokens = self.subword_tokenizer.tokenize(sequence_b)
            bert_input += sequence_b_sub_tokens + [self.SEP_TOKEN]

        if len(bert_input) > self.sequence_max_length:
            bert_input = bert_input[:self.sequence_max_length-1] + [self.SEP_TOKEN]

        token_type = utils.make_bert_token_type(bert_input, SEP_token=self.SEP_TOKEN)

        features = []
        features.append({
            "bert_input": bert_input,
            "token_type": {"feature": token_type, "text": ""},  # TODO: fix hard-code
        })

        return features, {}
