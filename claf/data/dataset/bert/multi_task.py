
import json
from overrides import overrides
import torch
import random

from claf.config.factory.data_loader import make_data_loader
from claf.data import utils
from claf.data.collate import PadCollator
from claf.data.dataset.base import DatasetBase


class MultiTaskBertDataset(DatasetBase):
    """
    Dataset for Multi-Task GLUE using BERT

    * Args:
        batch: Batch DTO (claf.data.batch)

    * Kwargs:
        helper: helper from data_reader
    """

    def __init__(self, batch, vocab, helper=None):
        super(MultiTaskBertDataset, self).__init__()

        self.name = "multitask_bert"
        self.vocab = vocab

        task_helpers = helper["task_helpers"]

        self.multi_dataset_size = 0
        self.task_datasets = []
        self.iterators = []
        for b, h in zip(batch, task_helpers):
            batch_size = h["batch_size"]
            dataset_cls = h["dataset"]
            dataset = dataset_cls(b, vocab, helper=h)
            data_loader = make_data_loader(dataset, batch_size=batch_size)  # TODO: cuda_device_id

            self.task_datasets.append(dataset)
            self.iterators.append(iter(data_loader))

            task_dataset_size, remain = divmod(len(dataset), batch_size)
            if remain > 0:
                task_dataset_size += 1
            self.multi_dataset_size += task_dataset_size

    @overrides
    def collate_fn(self, cuda_device_id=None):
        collator = PadCollator(cuda_device_id=cuda_device_id, pad_value=self.vocab.pad_index)

        def pass_tensor(data):
            print("data in collate_fn:", data)
            task_idx, tensor_datas = zip(*data)
            tensor_batch = tensor_datas[0]

            task_id_tensor = torch.LongTensor(list(task_idx))
            # task_id_tensor.cuda(cuda_device_id)
            tensor_batch.features["task_index"] = task_id_tensor
            print("task_idx:", task_idx[0], task_id_tensor)
            print("tensor_batch:", tensor_batch)
            return tensor_batch
        return pass_tensor

    @overrides
    def __getitem__(self, index):
        # self.lazy_evaluation(index)

        random_index = random.randint(0, len(self.iterators)-1)
        task_iterator = self.iterators[random_index]
        try:
            return random_index, next(task_iterator)
        except StopIteration as e:
            return self.__getitem__(index)

    def __len__(self):
        return self.multi_dataset_size

    def __repr__(self):
        dataset_properties = {
            "name": self.name,
            "total_count": self.__len__(),
            "dataset_count": len(self.iterators),
        }
        return json.dumps(dataset_properties, indent=4)

    def get_id(self, data_index):
        return self.data_ids[data_index]

    @overrides
    def get_ground_truth(self, data_id):
        return self.classes[data_id]

    def get_class_text_with_idx(self, class_index):
        if class_index is None:
            raise ValueError("class_index is required.")

        return self.class_idx2text[class_index]
