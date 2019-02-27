
import json
from collections import defaultdict
from overrides import overrides
import torch
from seqeval.metrics.sequence_labeling import get_entities

from claf.data import utils
from claf.data.collate import FeatLabelPadCollator
from claf.data.dataset.base import DatasetBase


class TokClsBertDataset(DatasetBase):
    """
    Dataset for Token Classification

    * Args:
        batch: Batch DTO (claf.data.batch)

    * Kwargs:
        helper: helper from data_reader
    """

    def __init__(self, batch, helper=None):
        super(TokClsBertDataset, self).__init__()

        self.name = "tok_cls_bert"
        self.helper = helper
        self.raw_dataset = helper["raw_dataset"]

        self.tag_idx2text = helper["tag_idx2text"]

        # Features
        self.bert_input_idx = [feature["bert_input"] for feature in batch.features]
        SEP_token = self.helper.get("sep_token", "[SEP]")
        self.token_type_idx = utils.make_bert_token_types(self.bert_input_idx, SEP_token=SEP_token)

        self.tagged_sub_token_idxs = [{"feature": feature["tagged_sub_token_idxs"]} for feature in batch.features]
        self.num_tokens = [{"feature": feature["num_tokens"]} for feature in batch.features]

        self.features = [self.bert_input_idx, self.token_type_idx]  # for lazy evaluation

        # Labels
        self.data_ids = {data_index: label["id"] for (data_index, label) in enumerate(batch.labels)}
        self.data_indices = list(self.data_ids.keys())

        self.tags = {
            label["id"]: {
                "tag_idxs": label["tag_idxs"],
                "tag_texts": label["tag_texts"],
            }
            for label in batch.labels
        }
        self.tag_texts = [label["tag_texts"] for label in batch.labels]
        self.tag_idxs = [label["tag_idxs"] for label in batch.labels]

        self.ignore_tag_idx = helper["ignore_tag_idx"]

    @overrides
    def collate_fn(self, cuda_device_id=None):
        """ collate: indexed features and labels -> tensor """
        collator = FeatLabelPadCollator(cuda_device_id=cuda_device_id)

        def make_tensor_fn(data):
            data_idxs, bert_input_idxs, token_type_idxs, tagged_token_idxs, num_tokens, tag_idxs_list = zip(*data)

            features = {
                "bert_input": utils.transpose(bert_input_idxs, skip_keys=["text"]),
                "token_type": utils.transpose(token_type_idxs, skip_keys=["text"]),
                "tagged_sub_token_idxs": utils.transpose(tagged_token_idxs, skip_keys=["text"]),
                "num_tokens": utils.transpose(num_tokens, skip_keys=["text"]),
            }
            labels = {
                "tag_idxs": tag_idxs_list,
                "data_idx": data_idxs,
            }
            return collator(
                features,
                labels,
                apply_pad_labels=["tag_idxs"],
                apply_pad_values=[self.ignore_tag_idx]
            )

        return make_tensor_fn

    @overrides
    def __getitem__(self, index):
        self.lazy_evaluation(index)

        return (
            self.data_indices[index],
            self.bert_input_idx[index],
            self.token_type_idx[index],
            self.tagged_sub_token_idxs[index],
            self.num_tokens[index],
            self.tag_idxs[index],
        )

    def __len__(self):
        return len(self.data_ids)

    def __repr__(self):
        dataset_properties = {
            "name": self.name,
            "total_count": self.__len__(),
            "num_tags": self.num_tags,
            "sequence_maxlen": self.sequence_maxlen,
            "tags": self.tag_idx2text,
        }
        return json.dumps(dataset_properties, indent=4)

    @property
    def num_tags(self):
        return len(self.tag_idx2text)

    @property
    def sequence_maxlen(self):
        return self._get_feature_maxlen(self.bert_input_idx)

    def get_id(self, data_index):
        return self.data_ids[data_index]

    @overrides
    def get_ground_truth(self, data_id):
        return self.tags[data_id]

    def get_tag_texts_with_idxs(self, tag_idxs):
        return [self.get_tag_text_with_idx(tag_idx)for tag_idx in tag_idxs]

    def get_tag_text_with_idx(self, tag_index):
        if tag_index is None:
            raise ValueError("tag_index is required.")

        return self.tag_idx2text[tag_index]
