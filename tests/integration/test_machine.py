
import json
import os
import pytest
import shutil

from claf.config.args import optimize_config, set_gpu_env
from claf.config.namespace import NestedNamespace
from claf.config.registry import Registry
from claf.learn.experiment import Experiment
from claf.learn.mode import Mode

import utils


TEST_DIR = os.path.join("logs", "test")
SQUAD_SYNTHETIC_DATA_PATH= os.path.join(TEST_DIR, "squad_synthetic_data.json")
WIKI_SYNTHETIC_DATA_PATH= os.path.join(TEST_DIR, "wiki_articles")


@pytest.mark.order1
def test_make_synthetic_data():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR, ignore_errors=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    utils.make_wiki_article_synthetic_data(WIKI_SYNTHETIC_DATA_PATH)
    utils.make_squad_synthetic_data(SQUAD_SYNTHETIC_DATA_PATH)


@pytest.fixture
def train_config(request):
    config_path = request.param

    config = NestedNamespace()
    with open(config_path, "r") as f:
        defined_config = json.load(f)
    config.load_from_json(defined_config)
    config.nsml = NestedNamespace()
    config.nsml.pause = 0
    config = optimize_config(config, is_test=True)
    set_gpu_env(config)

    config.data_reader.train_file_path = SQUAD_SYNTHETIC_DATA_PATH
    config.data_reader.valid_file_path = SQUAD_SYNTHETIC_DATA_PATH
    return config


@pytest.mark.order2
@pytest.mark.parametrize("train_config", ["./base_config/test/bidaf.json"], indirect=True)
def test_train_squad_bidaf_model(train_config):
    experiment = Experiment(Mode.TRAIN, train_config)
    experiment()


@pytest.fixture
def open_qa_config(request):
    config_path = request.param

    machine_config = NestedNamespace()
    with open(config_path, "r") as f:
        defined_config = json.load(f)
    machine_config.load_from_json(defined_config)

    claf_name = machine_config.name
    config = getattr(machine_config, claf_name, {})

    config.knowledge_base.wiki = WIKI_SYNTHETIC_DATA_PATH
    config.reasoning.reading_comprehension.checkpoint_path = "./logs/test/bidaf/checkpoint/model_1.pkl"
    return machine_config


@pytest.mark.order3
@pytest.mark.parametrize("open_qa_config", ["./base_config/test/open_qa.json"], indirect=True)
def test_open_qa_with_bidaf_model(open_qa_config):
    claf_name = open_qa_config.name
    config = getattr(open_qa_config, claf_name, {})

    registry = Registry()
    claf_machine = registry.get(f"machine:{claf_name}")(config)

    question = utils.make_random_tokens(5)
    answer = claf_machine(question)
    answer = json.dumps(answer, indent=4, ensure_ascii=False)
