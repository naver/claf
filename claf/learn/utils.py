
from collections import OrderedDict
import json
import logging
from pathlib import Path
import os
import re

import torch
from torch.nn import DataParallel
import requests

from claf import nsml
from claf.tokens.vocabulary import Vocab


logger = logging.getLogger(__name__)


""" Train Counter """


class TrainCounter:

    global_step = 0
    epoch = 0

    def __init__(self, display_unit="epoch"):
        if type(display_unit) == int:
            display_unit = f"every_{display_unit}_global_step"
        self.display_unit = display_unit

    def get_display(self):
        if self.display_unit == "epoch":
            return self.epoch
        else:
            return self.global_step


""" Save and Load checkpoint """


def load_model_checkpoint(model, checkpoint):
    model.load_state_dict(checkpoint["weights"])
    model.config = checkpoint["config"]
    model.metrics = checkpoint["metrics"]
    model.init_params = checkpoint["init_params"]
    model.predict_helper = checkpoint["predict_helper"]
    model.train_counter = checkpoint["train_counter"]
    model.vocabs = load_vocabs(checkpoint)

    logger.info(f"Load model checkpoints...!")
    return model


def load_optimizer_checkpoint(optimizer, checkpoint):
    optimizer.load_state_dict(checkpoint["optimizer"])

    logger.info(f"Load optimizer checkpoints...!")
    return optimizer


def load_vocabs(model_checkpoint):
    vocabs = {}
    token_config = model_checkpoint["config"]["token"]
    for token_name in token_config["names"]:
        token = token_config[token_name]
        vocab_config = token.get("vocab", {})

        texts = model_checkpoint["vocab_texts"][token_name]
        vocabs[token_name] = Vocab(token_name, **vocab_config).from_texts(texts)
    return vocabs


def save_checkpoint(path, model, optimizer, max_to_keep=10):
    path = Path(path)

    checkpoint_dir = path / "checkpoint"
    checkpoint_dir.mkdir(exist_ok=True)

    # Remove old checkpoints
    sorted_path = get_sorted_path(checkpoint_dir)
    if len(sorted_path) > max_to_keep:
        remove_train_counts = list(sorted_path.keys())[: -(max_to_keep - 1)]
        for train_count in remove_train_counts:
            optimizer_path = sorted_path[train_count].get("optimizer", None)
            if optimizer_path:
                os.remove(optimizer_path)

            model_path = sorted_path[train_count].get("model", None)
            if model_path:
                os.remove(model_path)

    train_counter = model.train_counter

    optimizer_path = checkpoint_dir / f"optimizer_{train_counter.get_display()}.pkl"
    torch.save({"optimizer": optimizer.state_dict()}, optimizer_path)

    model_path = checkpoint_dir / f"model_{train_counter.get_display()}.pkl"
    torch.save(
        {
            "config": model.config,
            "init_params": model.init_params,
            "predict_helper": model.predict_helper,
            "metrics": model.metrics,
            "train_counter": model.train_counter,
            "vocab_texts": {k: v.to_text() for k, v in model.vocabs.items()},
            "weights": model.state_dict(),
        },
        model_path,
    )

    # Write Vocab as text file (Only once)
    vocab_dir = path / "vocab"
    vocab_dir.mkdir(exist_ok=True)

    for token_name, vocab in model.vocabs.items():
        vocab_path = vocab_dir / f"{token_name}.txt"
        if not vocab_path.exists():
            vocab.dump(vocab_path)

    logger.info(f"Save {train_counter.global_step} global_step checkpoints...!")


def get_sorted_path(checkpoint_dir, both_exist=False):
    paths = []
    for root, dirs, files in os.walk(checkpoint_dir):
        for f_name in files:
            if "model" in f_name or "optimizer" in f_name:
                paths.append(Path(root) / f_name)

    path_with_train_count = {}
    for path in paths:
        train_count = re.findall("\d+", path.name)[0]
        train_count = int(train_count)
        if train_count not in path_with_train_count:
            path_with_train_count[train_count] = {}

        if "model" in path.name:
            path_with_train_count[train_count]["model"] = path
        if "optimizer" in path.name:
            path_with_train_count[train_count]["optimizer"] = path

    if both_exist:
        remove_keys = []
        for key, checkpoint in path_with_train_count.items():
            if not ("model" in checkpoint and "optimizer" in checkpoint):
                remove_keys.append(key)

        for key in remove_keys:
            del path_with_train_count[key]

    return OrderedDict(sorted(path_with_train_count.items()))


""" NSML """


def bind_nsml(model, **kwargs):  # pragma: no cover
    if type(model) == DataParallel:
        model = model.module

    def infer(raw_data, **kwargs):
        print("raw_data:", raw_data)

    def load(path, *args):
        checkpoint = torch.load(path)

        model.load_state_dict(checkpoint["weights"])
        model.config = checkpoint["config"]
        model.metrics = checkpoint["metrics"]
        model.init_params = checkpoint["init_params"],
        model.predict_helper = checkpoint["predict_helper"],
        model.train_counter = checkpoint["train_counter"]
        model.vocabs = load_vocabs(checkpoint)

        if "optimizer" in kwargs:
            kwargs["optimizer"].load_state_dict(checkpoint["optimizer"])
        logger.info(f"Load checkpoints...! {path}")

    def save(path, *args):
        # save the model with 'checkpoint' dictionary.
        checkpoint = {
            "config": model.config,
            "init_params": model.init_params,
            "predict_helper": model.predict_helper,
            "metrics": model.metrics,
            "train_counter": model.train_counter,
            "vocab_texts": {k: v.to_text() for k, v in model.vocabs.items()},
            "weights": model.state_dict(),
        }

        if "optimizer" in kwargs:
            checkpoint["optimizer"] = (kwargs["optimizer"].state_dict(),)

        torch.save(checkpoint, path)

        train_counter = model.train_counter
        logger.info(f"Save {train_counter.global_step} global_step checkpoints...! {path}")

    # function in function is just used to divide the namespace.
    nsml.bind(save, load, infer)


""" Notification """


def get_session_name():
    session_name = "local"
    if nsml.IS_ON_NSML:
        session_name = nsml.SESSION_NAME
    return session_name


def send_message_to_slack(webhook_url, title=None, message=None):  # pragma: no cover
    if message is None:
        data = {"text": f"{get_session_name()} session is exited."}
    else:
        data = {"attachments": [{"title": title, "text": message, "color": "#438C56"}]}

    try:
        if webhook_url == "":
            print(data["text"])
        else:
            requests.post(webhook_url, data=json.dumps(data))
    except Exception as e:
        print(str(e))
