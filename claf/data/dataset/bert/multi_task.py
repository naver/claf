
import json
from overrides import overrides
import torch
import random

from claf.config.factory.data_loader import make_data_loader
from claf.data import utils
from claf.data.dataset.base import DatasetBase


class MultiTaskBertDataset(DatasetBase):
    """
    Dataset for Multi-Task GLUE using BERT

    * Args:
        batch: Batch DTO (claf.data.batch)

    * Kwargs:
        helper: helper from data_reader
    """

    def __init__(self, batches, vocab, helper=None):
        super(MultiTaskBertDataset, self).__init__()

        self.name = "multitask_bert"
        self.vocab = vocab

        task_helpers = helper["task_helpers"]

        self.multi_dataset_size = 0
        self.batch_sizes = []
        self.task_datasets = []

        for b, h in zip(batches, task_helpers):
            batch_size = h["batch_size"]
            self.batch_sizes.append(batch_size)

            dataset_cls = h["dataset"]
            dataset = dataset_cls(b, vocab, helper=h)
            self.task_datasets.append(dataset)

            task_dataset_size, remain = divmod(len(dataset), batch_size)
            if remain > 0:
                task_dataset_size += 1
            self.multi_dataset_size += task_dataset_size

        self.init_iterators()

    def init_iterators(self):
        cuda_device_id = None
        if torch.cuda.is_available():
            cuda_device_id = 0  # Hard-code

        self.iterators = []
        for batch_size, dataset in zip(self.batch_sizes, self.task_datasets):
            data_loader = make_data_loader(dataset, batch_size=batch_size, cuda_device_id=cuda_device_id)  # TODO: cuda_device_id
            self.iterators.append(iter(data_loader))

        self.available_iterators = list(range(len(self.iterators)))

    @overrides
    def collate_fn(self, cuda_device_id=None):

        def pass_tensor(data):
            task_idx, tensor_datas = zip(*data)
            tensor_batch = tensor_datas[0]

            task_id_tensor = torch.LongTensor(list(task_idx))
            if torch.cuda.is_available():
                task_id_tensor.cuda(cuda_device_id)
            tensor_batch.features["task_index"] = task_id_tensor
            return tensor_batch
        return pass_tensor

    @overrides
    def __getitem__(self, index):
        # self.lazy_evaluation(index)
        if len(self.available_iterators) == 0:
            self.init_iterators()

        random_index = random.choice(self.available_iterators)
        task_iterator = self.iterators[random_index]
        try:
            return random_index, next(task_iterator)
        except StopIteration as e:
            self.available_iterators.remove(random_index)
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
