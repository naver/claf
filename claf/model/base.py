
import json
from pathlib import Path

import torch.nn as nn


class ModelBase(nn.Module):
    """
    Model Base Class

    Args:
        token_embedder: (claf.tokens.token_embedder.base) TokenEmbedder
    """

    def __init__(self):
        super(ModelBase, self).__init__()

    def forward(self, inputs):
        raise NotImplementedError

    def make_metrics(self, predictions):

        raise NotImplementedError

    def make_predictions(self, features):
        """
        for Metrics
        """

        raise NotImplementedError

    def predict(self, features):
        """
        Inference
        """

        raise NotImplementedError

    def print_examples(self, params):
        """
        Print evaluation examples
        """

        raise NotImplementedError

    def write_predictions(self, predictions, file_path=None, is_dict=True):
        data_type = "train" if self.training else "valid"

        pred_dir = Path(self._log_dir) / "predictions"
        pred_dir.mkdir(exist_ok=True)

        if file_path is None:
            file_path = f"predictions-{data_type}-{self._train_counter.get_display()}.json"

        pred_path = pred_dir / file_path
        with open(pred_path, "w") as out_file:
            if is_dict:
                out_file.write(json.dumps(predictions, indent=4))
            else:
                out_file.write(predictions)

    def is_ready(self):
        properties = [
            self._config,
            self._log_dir,
            # self._dataset,  It's set at _run_epoch()
            # self._metrics,  It's set at save()
            self._train_counter,
            self._vocabs
        ]

        return all([p is not None for p in properties])

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = config

    @property
    def log_dir(self):
        return self._log_dir

    @log_dir.setter
    def log_dir(self, log_dir):
        self._log_dir = log_dir

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        self._dataset = dataset

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        self._metrics = metrics

    @property
    def train_counter(self):
        return self._train_counter

    @train_counter.setter
    def train_counter(self, train_counter):
        self._train_counter = train_counter

    @property
    def vocabs(self):
        return self._vocabs

    @vocabs.setter
    def vocabs(self, vocabs):
        self._vocabs = vocabs


class ModelWithTokenEmbedder(ModelBase):
    def __init__(self, token_embedder):
        super(ModelWithTokenEmbedder, self).__init__()

        self.token_embedder = token_embedder
        if token_embedder is not None:
            self._vocabs = token_embedder.vocabs


class ModelWithoutTokenEmbedder(ModelBase):
    def __init__(self, token_makers):
        super(ModelWithoutTokenEmbedder, self).__init__()

        self.token_makers = token_makers
        self._vocabs = {
            token_name: token_maker.vocab for token_name, token_maker in token_makers.items()
        }
