
from torch.utils.data.dataset import Dataset

from claf.data import utils


class DatasetBase(Dataset):
    """
    Dataset Base Model
    An abstract class representing a Dataset.
    """

    def __init__(self):
        # Features - Lazy Evaluation
        self.f_count = 0
        self.features = []

    def __getitem__(self, index):
        raise NotImplementedError

    def _get_feature_maxlen(self, features):
        max_len = -1
        for feature in features:
            for token_name, sentence in feature.items():
                if token_name == "text":
                    continue
                if callable(sentence):
                    continue

                max_len = max(max_len, len(sentence))
        return max_len

    def collate_fn(self, cuda_device_id):
        raise NotImplementedError

    def get_ground_truths(self, data_idxs):
        data_idxs_dim = utils.get_token_dim(data_idxs)
        if data_idxs_dim > 2:
            raise ValueError(f"data_idxs dimension can't be larger than 2.({data_idxs_dim})")

        if data_idxs_dim == 2:
            return [self.get_ground_truth(data_id) for data_id in data_idxs]
        elif data_idxs_dim == 1:
            return self.get_ground_truth(data_idxs)
        else:
            raise ValueError(f"data_idxs dimension must be 1 or 2. not {data_idxs_dim}")

    def get_ground_truth(self):
        raise NotImplementedError

    def get_predict(self):
        raise NotImplementedError

    def lazy_evaluation(self, index):
        if self.f_count < self.__len__():
            self.f_count += 1

            for feature in self.features:
                for k, v in feature[index].items():
                    if utils.is_lazy(v):
                        feature[index][k] = v()
