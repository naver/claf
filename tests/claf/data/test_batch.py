
from claf.data.batch import Batch, make_batch


def test_make_batch():
    features = {
        "f1": 0,
        "f2": 1,
        "f3": 3,
    }

    labels = {
        "l1": 0,
        "l2": 1,
        "l3": 2,
    }

    batch = make_batch(features, labels)

    assert batch.features == features
    assert batch.labels == labels


def test_batch_sort_by_key():

    features = [
        {"f1": "long long long"},
        {"f1": "short"},
        {"f1": "mid mid"}
    ]

    labels = [
        {"l1": 3},
        {"l1": 1},
        {"l1": 2},
    ]

    batch = make_batch(features, labels)
    batch.sort_by_key("f1")

    assert batch.features == sorted(features, key=lambda x: len(x["f1"]))
