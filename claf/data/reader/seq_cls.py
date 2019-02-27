
import json
import logging
import uuid

from overrides import overrides
from tqdm import tqdm

from claf.data.dataset.seq_cls import SeqClsDataset
from claf.data.batch import make_batch
from claf.data.reader.base import DataReader
from claf.decorator import register

logger = logging.getLogger(__name__)


@register("reader:seq_cls")
class SeqClsReader(DataReader):
    """
    DataReader for Sequence Classification

    * Args:
        file_paths: .json file paths (train and dev)
        tokenizers: define tokenizers config (word)

    * Kwargs:
        class_key: name of the label in .json file to use for classification
    """

    def __init__(self, file_paths, tokenizers, sequence_max_length=None, class_key="class"):
        super(SeqClsReader, self).__init__(file_paths, SeqClsDataset)

        self.sequence_max_length = sequence_max_length
        self.text_columns = ["sequence"]

        if "word" not in tokenizers:
            raise ValueError("WordTokenizer is required. define WordTokenizer")

        self.word_tokenizer = tokenizers["word"]
        self.class_key = class_key

    def _get_data(self, file_path, **kwargs):
        data = self.data_handler.read(file_path)
        seq_cls_data = json.loads(data)

        return seq_cls_data, seq_cls_data["data"]

    def _get_class_dicts(self, **kwargs):
        seq_cls_data = kwargs["data"]

        class_idx2text = {class_idx: str(class_text) for class_idx, class_text in enumerate(seq_cls_data[self.class_key])}
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

            "model": {
                "num_classes": len(class_idx2text),
            },
            "predict_helper": {
                "class_idx2text": class_idx2text,
            }
        }
        features, labels = [], []

        for example in tqdm(raw_dataset, desc=data_type):
            sequence = example["sequence"].strip().replace("\n", "")
            sequence_words = self.word_tokenizer.tokenize(sequence)

            if (
                    self.sequence_max_length is not None
                    and data_type == "train"
                    and len(sequence_words) > self.sequence_max_length
            ):
                continue

            if "uid" in example:
                data_uid = example["uid"]
            else:
                data_uid = str(uuid.uuid1())

            feature_row = {
                "id": data_uid,
                "sequence": sequence,
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
                "sequence": sequence,
                "class_idx": class_text2idx[class_text],
                "class_text": class_text,
            }

        return make_batch(features, labels), helper

    def read_one_example(self, inputs):
        """ inputs keys: sequence """
        sequence = inputs["sequence"].strip().replace("\n", "")

        inputs["sequence"] = sequence

        return inputs, {}
