
import json
from overrides import overrides

from claf.data import utils
from claf.data.collate import PadCollator
from claf.data.dataset.base import DatasetBase


class RegressionBertDataset(DatasetBase):
    """
    Dataset for Regression using BERT

    * Args:
        batch: Batch DTO (claf.data.batch)

    * Kwargs:
        helper: helper from data_reader
    """

    def __init__(self, batch, vocab, helper=None):
        super(RegressionBertDataset, self).__init__()

        self.name = "reg_bert"
        self.vocab = vocab
        self.helper = helper

        # Features
        self.bert_input_idx = [feature["bert_input"] for feature in batch.features]
        SEP_token = self.helper.get("sep_token", "[SEP]")
        self.token_type_idx = utils.make_bert_token_types(self.bert_input_idx, SEP_token=SEP_token)

        self.features = [self.bert_input_idx, self.token_type_idx]  # for lazy evaluation

        # Labels
        self.data_ids = {data_index: label["id"] for (data_index, label) in enumerate(batch.labels)}
        self.data_indices = list(self.data_ids.keys())

        self.labels = {
            label["id"]: {
                "score": label["score"],
            }
            for label in batch.labels
        }

        self.label_scores = [label["score"] for label in batch.labels]

    @overrides
    def collate_fn(self, cuda_device_id=None):
        """ collate: indexed features and labels -> tensor """
        collator = PadCollator(cuda_device_id=cuda_device_id, pad_value=self.vocab.pad_index)

        def make_tensor_fn(data):
            data_idxs, bert_input_idxs, token_type_idxs, label_scores = zip(*data)

            features = {
                "bert_input": utils.transpose(bert_input_idxs, skip_keys=["text"]),
                "token_type": utils.transpose(token_type_idxs, skip_keys=["text"]),
            }
            labels = {
                "data_idx": data_idxs,
                "score": label_scores,
            }
            return collator(features, labels)

        return make_tensor_fn

    @overrides
    def __getitem__(self, index):
        self.lazy_evaluation(index)

        return (
            self.data_indices[index],
            self.bert_input_idx[index],
            self.token_type_idx[index],
            self.label_scores[index],
        )

    def __len__(self):
        return len(self.data_ids)

    def __repr__(self):
        dataset_properties = {
            "name": self.name,
            "total_count": self.__len__(),
            "sequence_maxlen": self.sequence_maxlen,
        }
        return json.dumps(dataset_properties, indent=4)

    @property
    def sequence_maxlen(self):
        return self._get_feature_maxlen(self.bert_input_idx)

    def get_id(self, data_index):
        return self.data_ids[data_index]

    @overrides
    def get_ground_truth(self, data_id):
        return self.labels[data_id]
