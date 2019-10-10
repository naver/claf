
import logging

from overrides import overrides

from claf.config.factory import DataReaderFactory
from claf.config.namespace import NestedNamespace
from claf.config.registry import Registry
from claf.data.dataset import MultiTaskBertDataset
from claf.data.dto import Helper
from claf.data.reader.base import DataReader
from claf.decorator import register
from claf.model.multi_task.category import TaskCategory

from .seq_cls import SeqClsBertReader
from .squad import SQuADBertReader
from .regression import RegressionBertReader
from .tok_cls import TokClsBertReader

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

        self.dataset_batches = []
        self.dataset_helpers = []
        self.tasks = []

        for reader in readers:
            data_reader = self.make_data_reader(reader)
            batches, helpers = data_reader.read()

            self.dataset_batches.append(batches)
            self.dataset_helpers.append(helpers)

            dataset_name = reader["dataset"]
            helper = helpers["train"]
            task = self.make_task_by_reader(dataset_name, data_reader, helper)
            self.tasks.append(task)

    def make_data_reader(self, config_dict):
        config = NestedNamespace()
        config.load_from_json(config_dict)
        config.tokenizers = self.tokenizers

        data_reader_factory = DataReaderFactory(config)
        return data_reader_factory.create()

    def make_task_by_reader(self, name, data_reader, helper):
        task = {}
        task["name"] = name
        task["metric_key"] = data_reader.METRIC_KEY

        if isinstance(data_reader, SeqClsBertReader):
            task["category"] = TaskCategory.SEQUENCE_CLASSIFICATION
            task["num_label"] = helper["model"]["num_classes"]
        elif isinstance(data_reader, SQuADBertReader):
            task["category"] = TaskCategory.READING_COMPREHENSION
            task["num_label"] = None
        elif isinstance(data_reader, RegressionBertReader):
            task["category"] = TaskCategory.REGRESSION
            task["num_label"] = 1
        elif isinstance(data_reader, TokClsBertReader):
            task["category"] = TaskCategory.TOKEN_CLASSIFICATION
            task["num_label"] = helper["model"]["num_tags"]
            task["ignore_tag_idx"] = helper["model"].get("ignore_tag_idx", 0)
        else:
            raise ValueError("Check data_reader.")

        return task

    @overrides
    def _read(self, file_path, data_type=None):
        """ TODO: Doc-String """

        batches = []
        helper = Helper()
        helper.task_helpers = []

        for b in self.dataset_batches:
            batches.append(b[data_type])
        for i, h in enumerate(self.dataset_helpers):
            task_helper = h[data_type]
            task_helper["batch_size"] = self.batch_sizes[i]

            helper.task_helpers.append(task_helper)

        helper.set_model_parameter({
            "tasks": self.tasks,
        })
        return batches, helper.to_dict()

    def read_one_example(self, inputs):
        pass
