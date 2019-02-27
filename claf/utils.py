
import logging
import os
import sys

from claf.learn.mode import Mode


""" Interface """


def get_user_input(category):
    print(f"{category.capitalize()} > ", end="")
    sys.stdout.flush()

    user_input = sys.stdin.readline()
    try:
        return eval(user_input)
    except BaseException:
        return str(user_input)


def flatten(l):
    for item in l:
        if isinstance(item, list):
            for in_item in flatten(item):
                yield in_item
        else:
            yield item


""" Logging """


def set_logging_config(mode, config):
    stdout_handler = logging.StreamHandler(sys.stdout)

    logging_handlers = [stdout_handler]
    logging_level = logging.INFO

    if mode == Mode.TRAIN:
        log_path = os.path.join(
            config.trainer.log_dir, f"{config.data_reader.dataset}_{config.model.name}.log"
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        logging_handlers.append(file_handler)
    elif mode == Mode.PREDICT:
        logging_level = logging.WARNING

    logging.basicConfig(
        format="%(asctime)s (%(filename)s:%(lineno)d): [%(levelname)s] - %(message)s",
        handlers=logging_handlers,
        level=logging_level,
    )
