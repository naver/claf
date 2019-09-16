
import logging

from overrides import overrides

from claf.config.factory import DataReaderFactory
from claf.config.namespace import NestedNamespace
from claf.config.registry import Registry
from claf.data.dataset import MultiTaskBertDataset
from claf.data.reader.base import DataReader
from claf.decorator import register

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
        readers=[]
    ):

        super(MultiTaskBertReader, self).__init__(file_paths, MultiTaskBertDataset)

        self.registry = Registry()

        self.text_columns = ["bert_input"]

        self.tokenizers = tokenizers

        self.dataset_features = []
        self.dataset_helpers = []

        for reader in readers:
            reader_config = NestedNamespace()
            reader_config.load_from_json(reader)
            reader_config.tokenizers = tokenizers

            data_reader_factory = DataReaderFactory(reader_config)
            data_reader = data_reader_factory.create()

            features, helpers = data_reader.read()

            self.dataset_features.append(features)
            self.dataset_helpers.append(helpers)

    @overrides
    def _read(self, file_path, data_type=None):
        """ TODO: Doc-String """

        features, helpers = [], []

        for f in self.dataset_features:
            features.append(f[data_type])
        for h in self.dataset_helpers:
            helpers.append(h[data_type])
        return features, helpers

    def read_one_example(self, inputs):
        pass
