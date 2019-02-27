
import json

from claf.config import args
from claf.config.namespace import NestedNamespace
from claf.learn.mode import Mode


def test_train_argparse():
    train_config = args.config(argv=["--seed_num", "4"], mode=Mode.TRAIN)

    assert train_config.seed_num == 4


def test_train_base_config_argparse():
    train_config = args.config(argv=["--base_config", "test/bidaf"], mode=Mode.TRAIN)

    config = NestedNamespace()
    with open("base_config/test/bidaf.json", "r") as f:
        defined_config = json.load(f)
    config.load_from_json(defined_config)
    args.set_gpu_env(config)

    assert train_config == config


def test_eval_argparse():
    eval_config = args.config(argv=["data_path", "checkpoint_path"], mode=Mode.EVAL)
    print(eval_config)


def test_predict_argparse():
    predict_config = args.config(argv=["checkpoint_path"], mode=Mode.PREDICT)
    print(predict_config)


def test_machine_argparse():
    machine_config = args.config(argv=["--machine_config", "ko_wiki"], mode=Mode.MACHINE)
    print(machine_config)
