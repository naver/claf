
import json
from overrides import overrides

import torch

from claf.data import utils
from claf.data.collate import PadCollator
from claf.data.dataset.base import DatasetBase



class WikiSQLDataset(DatasetBase):
    """
    WikiSQL Dataset

    * Args:
        batch: Batch DTO (claf.data.batch)

    * Kwargs:
        helper: helper from data_reader
    """

    def __init__(self, batch, helper=None):
        super(WikiSQLDataset, self).__init__()

        self.name = "wikisql"
        self.helper = helper

        # Features
        self.column_idx = [feature["column"] for feature in batch.features]
        self.question_idx = [feature["question"] for feature in batch.features]

        self.features = [self.column_idx, self.question_idx]

        # Labels
        self.data_idx = {data_index: label["id"] for (data_index, label) in enumerate(batch.labels)}
        self.data_indices = list(self.data_idx.keys())

        self.table_idx = {data_index: label["table_id"] for (data_index, label) in enumerate(batch.labels)}

        self.tokenized_question = {label["id"]: label["tokenized_question"] for label in batch.labels}

        self.labels = {
            label["id"]: {
                "agg_idx": label["aggregator_idx"],
                "sel_idx": label["select_column_idx"],
                "conds_num": label["conditions_num"],
                "conds_col": label["conditions_column_idx"],
                "conds_op": label["conditions_operator_idx"],
                "conds_val_str": label["conditions_value_string"],
                "conds_val_pos": label["conditions_value_position"],
                "sql_query": label["sql_query"],
                "execution_result": label["execution_result"],
            }
            for label in batch.labels
        }

    @overrides
    def collate_fn(self, cuda_device_id=None):
        """ collate: indexed features and labels -> tensor """
        collator = PadCollator(cuda_device_id=cuda_device_id)

        def make_tensor_fn(data):
            column_idxs, question_idxs, data_idxs = zip(*data)

            features = {
                "column": utils.transpose(column_idxs, skip_keys=["text"]),
                "question": utils.transpose(question_idxs, skip_keys=["text"]),
            }
            labels = {
                "data_idx": data_idxs,
            }
            return collator(features, labels)

        return make_tensor_fn

    @overrides
    def __getitem__(self, index):
        self.lazy_evaluation(index)

        return (
            self.column_idx[index],
            self.question_idx[index],
            self.data_indices[index],
        )

    def __len__(self):
        return len(self.data_idx)

    def __repr__(self):
        dataset_properties = {
            "name": self.name,
            "total_count": self.__len__(),
            "question_maxlen": self.question_maxlen,
        }
        return json.dumps(dataset_properties, indent=4)

    @property
    def question_maxlen(self):
        return self._get_feature_maxlen(self.question_idx)

    def get_id(self, data_index):
        if type(data_index) == torch.Tensor:
            data_index = data_index.item()
        return self.data_idx[data_index]

    def get_table_id(self, data_index):
        if type(data_index) == torch.Tensor:
            data_index = data_index.item()
        return self.table_idx[data_index]

    def get_tokenized_question(self, data_index):
        data_id = self.get_id(data_index)
        return self.tokenized_question[data_id]

    @overrides
    def get_ground_truth(self, data_index):
        if type(data_index) == torch.Tensor:
            data_id = self.get_id(data_index)
        else:
            data_id = data_index
        return self.labels[data_id]
