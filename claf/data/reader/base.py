
import logging

from claf.data.data_handler import CachePath, DataHandler
from claf import utils as common_utils

logger = logging.getLogger(__name__)


class DataReader:
    """
    DataReader Base Class

    * Args:
        file_paths: dictionary of consisting ('train' and 'vaild') file_path
        dataset_obj: Dataset Object (claf.data.dataset.base)
    """

    def __init__(self, file_paths, dataset_obj):
        self.file_paths = file_paths
        self.dataset_obj = dataset_obj

        self.data_handler = DataHandler(cache_path=CachePath.DATASET)  # for Concrete DataReader
        self.text_columns = None

    def filter_texts(self, dataset):
        texts = []

        def append_texts(datas):
            for data in datas:
                for key, value in data.items():
                    if key in self.text_columns:
                        texts.append(value)

        for data_type, dataset in dataset.items():
            append_texts(dataset.features)
            # append_texts(dataset.labels)

        texts = list(common_utils.flatten(texts))
        texts = list(set(texts))  # remove duplicate
        return texts

    def read(self):
        """ read with Concrete DataReader each type """

        if type(self.file_paths) != dict:
            raise ValueError(f"file_paths type is must be dict. not {type(self.file_paths)}")

        logger.info("Start read dataset")
        datasets, helpers = {}, {}
        for data_type, file_path in self.file_paths.items():
            if data_type is None:
                continue

            batch, helper = self._read(file_path, data_type=data_type)

            datasets[data_type] = batch
            helpers[data_type] = helper
        logger.info("Complete read dataset...\n")
        return datasets, helpers

    def _read(self, file_path, desc=None):
        raise NotImplementedError

    def read_one_example(self, inputs):
        helper = None
        return inputs, helper

    def convert_to_dataset(self, datas, helpers=None):
        """ Batch to Dataset """
        datasets = {}
        features, labels = [], []
        for k, batch in datas.items():
            if batch is None:
                continue
            datasets[k] = self.dataset_obj(batch, helper=helpers[k])
            logger.info(f"{k} dataset. {datasets[k]}")

            features += batch.features
            labels += batch.labels
        return datasets
