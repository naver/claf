
import json
import os
import pytest
import shutil

from claf.config.args import optimize_config, set_gpu_env
from claf.config.namespace import NestedNamespace
from claf.learn.experiment import Experiment
from claf.learn.mode import Mode



@pytest.fixture
def test_config(request):
    return load_and_setting(request.param)


def load_and_setting(config_path):
    config = NestedNamespace()
    with open(config_path, "r") as f:
        defined_config = json.load(f)
    config.load_from_json(defined_config)
    config.data_reader.wikisql = NestedNamespace()
    config.data_reader.wikisql.is_test = True
    config = optimize_config(config, is_test=True)
    set_gpu_env(config)
    return config


@pytest.mark.order1
@pytest.mark.parametrize("test_config", ["./base_config/test/sqlnet.json"], indirect=True)
def test_train_wikisql_sqlnet_model(test_config):
    os.system("sh script/download_wikisql.sh")

    experiment = Experiment(Mode.TRAIN, test_config)
    experiment()


@pytest.mark.order2
def test_qa_predict_wikisql_sqlnet_1_example():
    config = NestedNamespace()
    config.checkpoint_path = "./logs/test/sqlnet/checkpoint/model_1.pkl"
    config.cude_devices = None
    config.interactive = False
    set_gpu_env(config)

    config.column = ["Player", "No.", "Nationality", "Position", "Years in Toronto", "School/Club Team"]
    config.db_path = "data/wikisql/dev.db"
    config.table_id = "1-10015132-11"
    config.question = "What position does the player who played for butler cc (ks) play?"

    experiment = Experiment(Mode.PREDICT, config)
    experiment()


@pytest.mark.order3
def test_remove_tested_directory():
    test_path = "logs/test"
    shutil.rmtree(test_path)
