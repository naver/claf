# -*- coding: utf-8 -*-

import json

from claf.config import args
from claf.config.registry import Registry
from claf.learn.mode import Mode
from claf import utils as common_utils


if __name__ == "__main__":
    registry = Registry()

    machine_config = args.config(mode=Mode.MACHINE)
    machine_name = machine_config.name
    config = getattr(machine_config, machine_name, {})

    claf_machine = registry.get(f"machine:{machine_name}")(config)

    while True:
        question = common_utils.get_user_input(f"{getattr(machine_config, 'user_input', 'Question')}")
        answer = claf_machine.get_answer(question)
        answer = json.dumps(answer, indent=4, ensure_ascii=False)
        print(f"{getattr(machine_config, 'system_response', 'Answer')}: {answer}")
