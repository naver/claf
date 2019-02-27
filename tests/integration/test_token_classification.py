
import json
import os
import pytest
import shutil
import random

from claf.config.args import optimize_config, set_gpu_env
from claf.config.namespace import NestedNamespace
from claf.learn.experiment import Experiment
from claf.learn.mode import Mode

import utils


SYNTHETIC_DATA_PATH= os.path.join("logs", "test", "tok_cls", "synthetic_data.json")


@pytest.fixture
def test_config(request):
    return load_and_setting(request.param)


def load_and_setting(config_path):
    config = NestedNamespace()
    with open(config_path, "r") as f:
        defined_config = json.load(f)
    config.load_from_json(defined_config)
    config = optimize_config(config, is_test=True)
    set_gpu_env(config)

    config.data_reader.train_file_path = SYNTHETIC_DATA_PATH
    config.data_reader.valid_file_path = SYNTHETIC_DATA_PATH
    return config


@pytest.mark.order1
def test_make_synthetic_data():
    utils.make_tok_cls_synthetic_data(SYNTHETIC_DATA_PATH)


@pytest.mark.order2
@pytest.mark.parametrize("test_config", ["./base_config/test/bert_for_tok_cls.json"], indirect=True)
def test_train_bert_tok_cls_model(test_config):
    experiment = Experiment(Mode.TRAIN, test_config)
    experiment()


@pytest.mark.order3
def test_eval_nlu_bert_for_tok_cls():
    config = NestedNamespace()
    config.data_file_path = SYNTHETIC_DATA_PATH
    config.checkpoint_path = "./logs/test/tok_cls/bert/checkpoint/model_1.pkl"
    config.cude_devices = None
    set_gpu_env(config)

    experiment = Experiment(Mode.EVAL, config)
    experiment()


@pytest.mark.order3
def test_predict_nlu_bert_for_tok_cls_1_example():
    config = NestedNamespace()
    config.checkpoint_path = "./logs/test/tok_cls/bert/checkpoint/model_1.pkl"
    config.cude_devices = None
    config.interactive = False
    set_gpu_env(config)

    config.sequence = "hi, how are you?"

    experiment = Experiment(Mode.PREDICT, config)
    experiment()


@pytest.mark.order4
def test_remove_tested_directory():
    test_path = "logs/test"
    shutil.rmtree(test_path)
