
import json
import os
import pytest
import shutil

from claf.config.args import optimize_config, set_gpu_env
from claf.config.namespace import NestedNamespace
from claf.learn.experiment import Experiment
from claf.learn.mode import Mode

import utils


SYNTHETIC_DATA_PATH = os.path.join("logs", "test", "squad_synthetic_data.json")
DUMMY_EMBEDDING_300D_PATH = os.path.join("logs", "test", "dummy_300d.txt")


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
def test_make_squad_synthetic_data():
    utils.make_squad_synthetic_data(SYNTHETIC_DATA_PATH)
    utils.write_embedding_txt(DUMMY_EMBEDDING_300D_PATH, 300)


@pytest.mark.order2
@pytest.mark.parametrize("test_config", ["./base_config/test/bidaf.json"], indirect=True)
def test_train_squad_bidaf_model(test_config):
    experiment = Experiment(Mode.TRAIN, test_config)
    experiment()


@pytest.mark.order2
@pytest.mark.parametrize("test_config", ["./base_config/test/bidaf_no_answer.json"], indirect=True)
def test_train_squad_bidaf_no_answer_model(test_config):
    experiment = Experiment(Mode.TRAIN, test_config)
    experiment()


# need glove.840B.300d.txt (5.65 GB)
# @pytest.mark.order2
# @pytest.mark.parametrize("test_config", ["./base_config/test/bidaf+cove.json"], indirect=True)
# def test_train_squad_bidaf_cove_model(test_config):
    # experiment = Experiment(Mode.TRAIN, test_config)
    # experiment()


@pytest.mark.order2
@pytest.mark.parametrize("test_config", ["./base_config/test/bidaf+elmo.json"], indirect=True)
def test_train_squad_bidaf_elmo_model(test_config):
    experiment = Experiment(Mode.TRAIN, test_config)
    experiment()


@pytest.mark.order2
@pytest.mark.parametrize("test_config", ["./base_config/test/drqa.json"], indirect=True)
def test_train_squad_drqa_model(test_config):
    experiment = Experiment(Mode.TRAIN, test_config)
    experiment()


@pytest.mark.order2
@pytest.mark.parametrize("test_config", ["./base_config/test/drqa_sparse_to_embedding.json"], indirect=True)
def test_train_squad_drqa_model_with_sparse_to_embedding(test_config):
    experiment = Experiment(Mode.TRAIN, test_config)
    experiment()


@pytest.mark.order2
@pytest.mark.parametrize("test_config", ["./base_config/test/docqa.json"], indirect=True)
def test_train_squad_docqa_model(test_config):
    experiment = Experiment(Mode.TRAIN, test_config)
    experiment()


@pytest.mark.order2
@pytest.mark.parametrize("test_config", ["./base_config/test/docqa_no_answer.json"], indirect=True)
def test_train_squad_docqa_no_answer_model(test_config):
    experiment = Experiment(Mode.TRAIN, test_config)
    experiment()


@pytest.mark.order2
@pytest.mark.parametrize("test_config", ["./base_config/test/qanet.json"], indirect=True)
def test_train_squad_qanet_model(test_config):
    experiment = Experiment(Mode.TRAIN, test_config)
    experiment()


@pytest.mark.order2
@pytest.mark.parametrize("test_config", ["./base_config/test/bert_for_qa.json"], indirect=True)
def test_train_squad_bert_model(test_config):
    experiment = Experiment(Mode.TRAIN, test_config)
    experiment()


# TODO: subword ---> word
# @pytest.mark.order2
# @pytest.mark.parametrize("test_config", ["./base_config/test/bidaf+bert.json"], indirect=True)
# def test_train_squad_bidaf_model_with_bert(test_config):
    # experiment = Experiment(Mode.TRAIN, test_config)
    # experiment()


@pytest.mark.order2
def test_eval_squad_bidaf():
    config = NestedNamespace()
    config.data_file_path = SYNTHETIC_DATA_PATH
    config.checkpoint_path = "./logs/test/bidaf/checkpoint/model_1.pkl"
    config.cude_devices = None
    set_gpu_env(config)

    experiment = Experiment(Mode.EVAL, config)
    experiment()


@pytest.mark.order3
def test_eval_infer_squad_bidaf():
    config = NestedNamespace()
    config.data_file_path = SYNTHETIC_DATA_PATH
    config.checkpoint_path = "./logs/test/bidaf/checkpoint/model_1.pkl"
    config.cude_devices = None
    config.inference_latency = 1000
    set_gpu_env(config)

    experiment = Experiment(Mode.INFER_EVAL, config)
    experiment()


@pytest.mark.order3
def test_qa_predict_squad_bidaf_1_example():
    config = NestedNamespace()
    config.checkpoint_path = "./logs/test/bidaf/checkpoint/model_1.pkl"
    config.cude_devices = None
    config.interactive = False
    set_gpu_env(config)

    config.context = "Westwood One will carry the game throughout North America, with Kevin Harlan as play-by-play announcer, Boomer Esiason and Dan Fouts as color analysts, and James Lofton and Mark Malone as sideline reporters. Jim Gray will anchor the pre-game and halftime coverage."
    config.question = "What radio network carried the Super Bowl?"

    experiment = Experiment(Mode.PREDICT, config)
    experiment()


@pytest.mark.order3
def test_qa_predict_squad_bert_short_1_example():
    config = NestedNamespace()
    config.checkpoint_path = "./logs/test/bert_for_qa/checkpoint/model_1.pkl"
    config.cude_devices = None
    config.interactive = False
    set_gpu_env(config)

    config.context = "Westwood One will carry the game throughout North America, with Kevin Harlan as play-by-play announcer, Boomer Esiason and Dan Fouts as color analysts, and James Lofton and Mark Malone as sideline reporters. Jim Gray will anchor the pre-game and halftime coverage."
    config.question = "What radio network carried the Super Bowl?"

    experiment = Experiment(Mode.PREDICT, config)
    experiment()


@pytest.mark.order3
def test_qa_predict_squad_bert_long_1_example():
    config = NestedNamespace()
    config.checkpoint_path = "./logs/test/bert_for_qa/checkpoint/model_1.pkl"
    config.cude_devices = None
    config.interactive = False
    set_gpu_env(config)

    config.context = "hi ho hi ho 1 hi ho hi ho 2 hi ho hi ho 3 hi ho hi ho 4 hi ho hi ho 5 hi ho hi ho 6 hi ho hi ho 7 hi ho hi ho 8 hi ho hi ho hi 9 ho hi ho hi ho hi 10 ho hi ho hi ho hi ho 11 hi ho hi ho hi 12 ANSWER ho hi ho hi ho hi 13 ho hi ho hi ho hi 14 ho hi ho hi ho hi 15 ho hi ho hi ho hi 16 ho hi ho hi ho hi 17 ho hi ho hi ho hi 18 ho hi ho hi ho hi 19 ho hi ho hi ho hi 20 ho hi ho hi ho hi 21 ho hi ho hi ho hi 22 ho hi ho hi ho hi 23 ho hi ho hi ho hi 24 ho hi ho hi 25 ho hi ho hi ho 1 hi ho hi ho 2 hi ho hi ho 3 hi ho hi ho 4 hi ho hi ho 5 hi ho hi ho 6 hi ho hi ho 7 hi ho hi ho 8 hi ho hi ho hi 9 ho hi ho hi ho hi 10 ho hi ho hi ho hi ho 11 hi ho hi ho hi 12 ho hi ho hi ho hi 13 ho hi ho hi ho hi 14 ho hi ho hi ho hi 15 ho hi ho hi ho hi 16 ho hi ho hi ho hi 17 ho hi ho hi ho hi 18 ho hi ho hi ho hi 19 ho hi ho hi ho hi 20 ho hi ho hi ho hi 21 ho hi ho hi ho hi 22 ho hi ho hi ho hi 23 ho hi ho hi ho hi 24 ho hi ho hi 25 ho hi ho hi ho 1 hi ho hi ho 2 hi ho hi ho 3 hi ho hi ho 4 hi ho hi ho 5 hi ho hi ho 6 hi ho hi ho 7 hi ho hi ho 8 hi ho hi ho hi 9 ho hi ho hi ho hi 10 ho hi ho hi ho hi ho 11 hi ho hi ho hi 12 ho hi ho hi ho hi 13 ho hi ho hi ho hi 14 ho hi ho hi ho hi 15 ho hi ho hi ho hi 16 ho hi ho hi ho hi 17 ho hi ho hi ho hi 18 ho hi ho hi ho hi 19 ho hi ho hi ho hi 20 ho hi ho hi ho hi 21 ho hi ho hi ho hi 22 ho hi ho hi ho hi 23 ho hi ho hi ho hi 24 ho hi ho hi 25 ho hi ho hi ho 1 hi ho hi ho 2 hi ho hi ho 3 hi ho hi ho 4 hi ho hi ho 5 hi ho hi ho 6 hi ho hi ho 7 hi ho hi ho 8 hi ho hi ho hi 9 ho hi ho hi ho hi 10 ho hi ho hi ho hi ho 11 hi ho hi ho hi 12 ho hi ho hi ho hi 13 ho hi ho hi ho hi 14 ho hi ho hi ho hi 15 ho hi ho hi ho hi 16 ho hi ho hi ho hi 17 ho hi ho hi ho hi 18 ho hi ho hi ho hi 19 ho hi ho hi ho hi 20 ho hi ho hi ho hi 21 ho hi ho hi ho hi 22 ho hi ho hi ho hi 23 ho hi ho hi ho hi 24 ho hi ho hi 25 ho"
    config.question = "good hi ho hi ho hi good hi ho hi ho hi good hi ho hi ho hi good hi ho hi ho hi good hi ho hi ho hi"

    experiment = Experiment(Mode.PREDICT, config)
    experiment()


@pytest.mark.order4
def test_remove_tested_directory():
    test_path = "logs/test"
    shutil.rmtree(test_path)
