
import json
import logging
import uuid

from overrides import overrides
from tqdm import tqdm

from claf.data.dataset import SeqClsBertDataset
from claf.data.batch import make_batch
from claf.data.reader.base import DataReader
from claf.decorator import register
from claf.data import utils

logger = logging.getLogger(__name__)


@register("reader:seq_cls_bert")
class SeqClsBertReader(DataReader):
    """
    DataReader for Sequence Classification using BERT

    * Args:
        file_paths: .json file paths (train and dev)
        tokenizers: define tokenizers config (subword)

    * Kwargs:
        class_key: name of the label in .json file to use for classification
    """

    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    UNK_TOKEN = "[UNK]"
    CONTINUE_SYMBOL = "##"

    def __init__(
            self,
            file_paths,
            tokenizers,
            sequence_max_length=None,
            class_key="class"
    ):

        super(SeqClsBertReader, self).__init__(file_paths, SeqClsBertDataset)

        self.sequence_max_length = sequence_max_length
        self.text_columns = ["bert_input", "sequence"]

        if "subword" not in tokenizers:
            raise ValueError("WordTokenizer and SubwordTokenizer is required.")

        self.subword_tokenizer = tokenizers["subword"]

        self.class_key = class_key

    def _get_data(self, file_path, **kwargs):
        data = self.data_handler.read(file_path)
        seq_cls_data = json.loads(data)

        return seq_cls_data, seq_cls_data["data"]

    def _get_class_dicts(self, **kwargs):
        seq_cls_data = kwargs["data"]

        class_idx2text = {
            class_idx: str(class_text)
            for class_idx, class_text in enumerate(seq_cls_data[self.class_key])
        }
        class_text2idx = {class_text: class_idx for class_idx, class_text in class_idx2text.items()}

        return class_idx2text, class_text2idx

    @overrides
    def _read(self, file_path, data_type=None):
        """
        .json file structure should be something like this:

        {
            "data": [
                {
                    "sequence": "what a wonderful day!",
                    "emotion": "happy"
                },
                ...
            ],
            "emotion": [  // class_key
                "angry",
                "happy",
                "sad",
                ...
            ]
        }
        """

        data, raw_dataset = self._get_data(file_path, data_type=data_type)
        class_idx2text, class_text2idx = self._get_class_dicts(data=data)

        helper = {
            "file_path": file_path,
            "examples": {},
            "raw_dataset": raw_dataset,
            "class_idx2text": class_idx2text,
            "class_text2idx": class_text2idx,
            "cls_token": self.CLS_TOKEN,
            "sep_token": self.SEP_TOKEN,
            "unk_token": self.UNK_TOKEN,
            "continue_symbol": self.CONTINUE_SYMBOL,

            "model": {
                "num_classes": len(class_idx2text),
            },
            "predict_helper": {
                "class_idx2text": class_idx2text,
            }
        }
        features, labels = [], []

        for example in tqdm(raw_dataset, desc=data_type):
            sequence_text = example["sequence"].strip().replace("\n", "")
            sequence_sub_tokens = self.subword_tokenizer.tokenize(sequence_text)

            bert_input = [self.CLS_TOKEN] + sequence_sub_tokens + [self.SEP_TOKEN]

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

            class_text = example[self.class_key]
            label_row = {
                "id": data_uid,
                "class_idx": class_text2idx[class_text],
                "class_text": class_text,
            }
            labels.append(label_row)

            helper["examples"][data_uid] = {
                "sequence": sequence_text,
                "sequence_sub_tokens": sequence_sub_tokens,
                "class_idx": class_text2idx[class_text],
                "class_text": class_text,
            }

        return make_batch(features, labels), helper

    def read_one_example(self, inputs):
        """ inputs keys: sequence """
        sequence_text = inputs["sequence"].strip().replace("\n", "")
        sequence_sub_tokens = self.subword_tokenizer.tokenize(sequence_text)

        if len(sequence_sub_tokens) > self.sequence_max_length:
            sequence_sub_tokens = sequence_sub_tokens[:self.sequence_max_length]

        bert_input = [self.CLS_TOKEN] + sequence_sub_tokens + [self.SEP_TOKEN]
        token_type = utils.make_bert_token_type(bert_input, SEP_token=self.SEP_TOKEN)

        features = []
        features.append({
            "bert_input": bert_input,
            "token_type": {"feature": token_type, "text": ""},  # TODO: fix hard-code
        })

        return features, {}
