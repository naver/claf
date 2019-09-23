
import logging

from overrides import overrides

from claf.config.factory import DataReaderFactory
from claf.config.namespace import NestedNamespace
from claf.config.registry import Registry
from claf.data.dataset import MultiTaskBertDataset
from claf.data.dto import Helper
from claf.data.reader.base import DataReader
from claf.decorator import register

from .seq_cls import SeqClsBertReader
from .regression import RegressionBertReader

logger = logging.getLogger(__name__)


@register("reader:multitask_bert")
class MultiTaskBertReader(DataReader):
    """
    DataReader for Multi-Task using BERT

    * Args:
        file_paths: .json file paths (train and dev)
        tokenizers: define tokenizers config (subword)

    * Kwargs:
        class_key: name of the label in .json file to use for classification
    """

    CLASS_DATA = None

    def __init__(
        self,
        file_paths,
        tokenizers,
        batch_sizes=[],
        readers=[],
    ):

        super(MultiTaskBertReader, self).__init__(file_paths, MultiTaskBertDataset)
        assert len(batch_sizes) == len(readers)

        self.registry = Registry()

        self.text_columns = ["bert_input"]

        self.tokenizers = tokenizers
        self.batch_sizes = batch_sizes

        self.dataset_features = []
        self.dataset_helpers = []
        self.tasks = []

        for reader in readers:
            reader_config = NestedNamespace()
            reader_config.load_from_json(reader)
            reader_config.tokenizers = tokenizers

            data_reader_factory = DataReaderFactory(reader_config)
            data_reader = data_reader_factory.create()

            features, helpers = data_reader.read()

            self.dataset_features.append(features)
            self.dataset_helpers.append(helpers)

            task = {}
            if isinstance(data_reader, SeqClsBertReader):
                task["category"] = "classification"
                task["num_label"] = helpers["train"]["model"]["num_classes"]
            elif isinstance(data_reader, RegressionBertReader):
                task["category"] = "regression"
                task["num_label"] = 1

            self.tasks.append(task)

    @overrides
    def _read(self, file_path, data_type=None):
        """ TODO: Doc-String """

        features = []
        helper = Helper()
        helper.task_helpers = []

        for f in self.dataset_features:
            features.append(f[data_type])
        for i, h in enumerate(self.dataset_helpers):
            task_helper = h[data_type]
            task_helper["batch_size"] = self.batch_sizes[i]

            helper.task_helpers.append(task_helper)

        # TODO: tasks (category - issubclasses, label_count - num_classes or labl)
        helper.set_model_parameter({
            "tasks": self.tasks,
        })

        return features, helper.to_dict()

    def read_one_example(self, inputs):
        pass
