
import json
import os
import pytest

from claf.config.args import optimize_config, set_gpu_env
from claf.config.namespace import NestedNamespace
from claf.learn.experiment import Experiment
from claf.learn.mode import Mode

import utils


SYNTHETIC_QA_DATA_PATH = os.path.join("logs", "test", "data", "qa_synthetic_data.json")
SYNTHETIC_SEQ_CLS_DATA_PATH = os.path.join("logs", "test", "data", "seq_cls_synthetic_data.json")
SYNTHETIC_REG_DATA_PATH = os.path.join("logs", "test", "data", "reg_synthetic_data.json")


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

    config.data_reader.multitask_bert.readers[0]["train_file_path"] = SYNTHETIC_SEQ_CLS_DATA_PATH
    config.data_reader.multitask_bert.readers[0]["valid_file_path"] = SYNTHETIC_SEQ_CLS_DATA_PATH

    config.data_reader.multitask_bert.readers[1]["train_file_path"] = SYNTHETIC_REG_DATA_PATH
    config.data_reader.multitask_bert.readers[1]["valid_file_path"] = SYNTHETIC_REG_DATA_PATH

    config.data_reader.multitask_bert.readers[2]["train_file_path"] = SYNTHETIC_QA_DATA_PATH
    config.data_reader.multitask_bert.readers[2]["valid_file_path"] = SYNTHETIC_QA_DATA_PATH

    return config


@pytest.mark.order1
def test_make_multi_task_synthetic_data():
    utils.make_bert_seq_cls_synthetic_data(SYNTHETIC_SEQ_CLS_DATA_PATH, remove_exist=False)
    utils.make_bert_reg_synthetic_data(SYNTHETIC_REG_DATA_PATH, remove_exist=False)
    utils.make_squad_synthetic_data(SYNTHETIC_QA_DATA_PATH, remove_exist=False)


@pytest.mark.order2
@pytest.mark.parametrize("test_config", ["./base_config/test/bert_for_multi_task.json"], indirect=True)
def test_train_multi_task_bert_model(test_config):
    experiment = Experiment(Mode.TRAIN, test_config)
    experiment()
