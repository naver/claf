
import logging

logger = logging.getLogger(__name__)


def make_batch(features, labels):
    return Batch(**{"features": features, "labels": labels})


class Batch:
    """
    Batch Data Transfer Object (DTO) Class

    dictionary consisting of
        - features: (dict) input
        - labels: (dict) output
    """

    def __init__(self, **kwargs):
        if set(kwargs.keys()) != set(["features", "labels"]):
            raise ValueError("You can use only 'features' and 'labels' as dictionary key.")
        self.__dict__ = kwargs

    def __repr__(self):
        return str(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def sort_by_key(self, sort_key):
        logger.info(f"Start sort by key: {sort_key}'s length")

        zipped = zip(self.__dict__["features"], self.__dict__["labels"])

        features = self.__dict__["features"]
        if type(features) == list:
            feature_keys = list(features[0].keys())
        else:
            feature_keys = features.keys()

        key_index = 0 if sort_key in feature_keys else 1  # sort_key in features or labels

        sorted_features, sorted_labels = [], []
        for data in sorted(zipped, key=lambda x: len(x[key_index][sort_key])):
            feature, label = data
            sorted_features.append(feature)
            sorted_labels.append(label)

        self.__dict__["features"] = sorted_features
        self.__dict__["labels"] = sorted_labels
        zipped = None
        logger.info("Complete sorting...")

    def to_dict(self, flatten=False, recursive=True):
        def _flatten(d):
            if d == {}:
                return d

            k, v = d.popitem()
            if isinstance(v, dict):
                flat_v = _flatten(v)
                for f_k in list(flat_v.keys()):
                    flat_v[k + "#" + f_k] = flat_v[f_k]
                    del flat_v[f_k]
                return {**flat_v, **_flatten(d)}
            else:
                return {k: v, **_flatten(d)}

        def _recursive(d):
            if not isinstance(d, dict):
                return d

            for k, v in d.items():
                if isinstance(v, dict):
                    dict_v = dict(v)
                    d[k] = _recursive(dict_v)
            return d

        if flatten:
            d = {}
            d.update(_flatten(self.__dict__["features"]))
            d.update(_flatten(self.__dict__["labels"]))
            return d

        if recursive:
            return _recursive(self.__dict__)

        return dict(self.__dict__)
