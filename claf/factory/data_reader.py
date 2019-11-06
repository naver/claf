
from overrides import overrides

from claf.config.registry import Registry

from .base import Factory


class DataReaderFactory(Factory):
    """
    DataReader Factory Class

    Create Concrete reader according to config.dataset
    Get reader from reader registries (eg. @register("reader:{reader_name}"))

    * Args:
        config: data_reader config from argument (config.data_reader)
    """

    def __init__(self):
        self.registry = Registry()

    @overrides
    def create(self, config):
        dataset_name = config.dataset

        file_paths = {}
        if getattr(config, "train_file_path", None):
            file_paths["train"] = config.train_file_path
        if getattr(config, "valid_file_path", None):
            file_paths["valid"] = config.valid_file_path

        reader_config = {"file_paths": file_paths}
        if "params" in config and type(config.params) == dict:
            reader_config.update(config.params)
        if "tokenizers" in config:
            reader_config["tokenizers"] = config.tokenizers

        dataset_config = getattr(config, config.dataset, None)
        if dataset_config is not None:
            dataset_config = vars(dataset_config)
            reader_config.update(dataset_config)

        reader = self.registry.get(f"reader:{dataset_name.lower()}")
        return reader(**reader_config)
