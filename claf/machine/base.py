
from argparse import Namespace
import json

from claf.config.namespace import NestedNamespace
from claf.config.registry import Registry
from claf.learn.experiment import Experiment
from claf.learn.mode import Mode
from claf.machine.module import Module


class Machine:
    """
    Machine: Combine modules then make a NLP Machine

    * Args:
        config: machine_config
    """

    def __init__(self, config):
        self.config = config
        self.registry = Registry()

    def load(self):
        raise NotImplementedError("")

    @classmethod
    def load_from_config(cls, config_path):
        with open(config_path, "r", encoding="utf-8") as in_file:
            machine_config = NestedNamespace()
            machine_config.load_from_json(json.load(in_file))

        machine_name = machine_config.name
        config = getattr(machine_config, machine_name, {})
        return cls(config)

    def __call__(self, text):
        raise NotImplementedError("")

    def make_module(self, config):
        """
        Make component or experiment for claf Machine's module

        * Args:
            - config: module's config (claf.config.namespace.NestedNamespace)
        """

        module_type = config.type
        if module_type == Module.COMPONENT:
            name = config.name
            module_config = getattr(config, name, {})
            if isinstance(module_config, Namespace):
                module_config = vars(module_config)

            if getattr(config, "params", None):
                module_config.update(config.params)
            return self.registry.get(f"component:{name}")(**module_config)
        elif module_type == Module.EXPERIMENT:
            experiment_config = Namespace()
            experiment_config.checkpoint_path = config.checkpoint_path
            experiment_config.cuda_devices = getattr(config, "cuda_devices", None)
            experiment_config.interactive = False

            experiment = Experiment(Mode.PREDICT, experiment_config)
            experiment.set_predict_mode(preload=True)
            return experiment
        else:
            raise ValueError(
                f"module_type is available only [component|experiment]. not '{module_type}'"
            )
