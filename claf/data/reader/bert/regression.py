
import logging
import json
import uuid

from overrides import overrides
from tqdm import tqdm

from claf.data.dataset import RegressionBertDataset
from claf.data.dto import BertFeature, Helper
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

    def __init__(
        self,
        file_paths,
        tokenizers,
        sequence_max_length=None,
        label_key="score",
        cls_token="[CLS]",
        sep_token="[SEP]",
        input_type="bert",
        is_test=False,
    ):

        super(RegressionBertReader, self).__init__(file_paths, RegressionBertDataset)

        self.sequence_max_length = sequence_max_length
        self.text_columns = ["bert_input", "sequence"]

        # Tokenizers
        # - BERT: Word + Subword or Word + Char
        # - RoBERTa: BPE

        if input_type == "bert":
            self.tokenizer = tokenizers.get("subword", None)
            if self.tokenizer is None:
                self.tokenizer["char"]
        elif input_type == "roberta":
            self.tokenizer = tokenizers["bpe"]
        else:
            raise ValueError("'bert' and 'roberta' are available input_type.")

        self.label_key = label_key
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.input_type = input_type
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

        helper = Helper(**{
            "file_path": file_path,
            "cls_token": self.cls_token,
            "sep_token": self.sep_token,
        })

        features, labels = [], []

        for example in tqdm(data, desc=data_type):
            sequence_a = utils.get_sequence_a(example)
            sequence_b = example.get("sequence_b", None)

            sequence_a_tokens = self.tokenizer.tokenize(sequence_a)
            sequence_b_tokens = None
            if sequence_b:
                sequence_b_tokens = self.tokenizer.tokenize(sequence_b)

            bert_input = utils.make_bert_input(
                sequence_a,
                sequence_b,
                self.tokenizer,
                max_seq_length=self.sequence_max_length,
                data_type=data_type,
                cls_token=self.cls_token,
                sep_token=self.sep_token,
                input_type=self.input_type,
            )

            if bert_input is None:
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

            helper.set_example(data_uid, {
                "sequence_a": sequence_a,
                "sequence_a_tokens": sequence_a_tokens,
                "sequence_b": sequence_b,
                "sequence_b_tokens": sequence_b_tokens,
                "score": score,
            })

            if self.is_test and len(features) >= 10:
                break

        return utils.make_batch(features, labels), helper.to_dict()

    def read_one_example(self, inputs):
        """ inputs keys: sequence_a and sequence_b """
        sequence_a = utils.get_sequence_a(inputs)
        sequence_b = inputs.get("sequence_b", None)

        bert_feature = BertFeature()
        bert_feature.set_input_with_speical_token(
            sequence_a,
            sequence_b,
            self.tokenizer,
            max_seq_length=self.sequence_max_length,
            data_type="predict",
            cls_token=self.cls_token,
            sep_token=self.sep_token,
            input_type=self.input_type,
        )

        features = [bert_feature.to_dict()]
        helper = {}
        return features, helper
