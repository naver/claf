
from overrides import overrides
import torch

from claf.config.namespace import NestedNamespace
from claf.learn.optimization.learning_rate_scheduler import get_lr_schedulers
from claf.learn.optimization.learning_rate_scheduler import (
    LearningRateWithoutMetricsWrapper,
    LearningRateWithMetricsWrapper,
)
from claf.learn.optimization.optimizer import get_optimizer_by_name
from claf.model.sequence_classification import BertForSeqCls, RobertaForSeqCls

from .base import Factory


class OptimizerFactory(Factory):
    """
    Optimizer Factory Class

    include optimizer, learning_rate_scheduler and exponential_moving_average

    * Args:
        config: optimizer config from argument (config.optimizer)
    """

    def __init__(self):
        pass

    @overrides
    def create(self, config, model):

        if not issubclass(type(model), torch.nn.Module):
            raise ValueError("optimizer model is must be subclass of torch.nn.Module.")

        # Optimizer
        op_type = config.op_type
        optimizer_params = {"lr": config.learning_rate}

        op_config = getattr(config, op_type, None)
        if op_config is not None:
            op_config = vars(op_config)
            optimizer_params.update(op_config)

        model_parameters = self.get_model_parameters(model, optimizer_params)
        optimizer = get_optimizer_by_name(op_type)(model_parameters, **optimizer_params)
        op_dict = {"optimizer": optimizer}

        # LearningRate Scheduler
        lr_scheduler = self.make_lr_scheduler(config, optimizer)
        if lr_scheduler is not None:
            op_dict["learning_rate_scheduler"] = lr_scheduler

        # exponential_moving_average
        ema_value = getattr(config, "exponential_moving_average", None)
        if ema_value and ema_value > 0:
            op_dict["exponential_moving_average"] = ema_value

        return op_dict

    def get_model_parameters(self, model, optimizer_params):
        if getattr(model, "use_pytorch_transformers", False):
            weight_decay = optimizer_params.get("weight_decay", 0)
            model_parameters = self._group_parameters_for_transformers(model, weight_decay=weight_decay)
        else:
            model_parameters = [param for param in model.parameters() if param.requires_grad]
        return model_parameters

    def _group_parameters_for_transformers(self, model, weight_decay=0):
        # Prepare optimizer
        param_optimizer = list(model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        if not isinstance(model, BertForSeqCls) or not isinstance(model, RobertaForSeqCls):
            param_optimizer = [n for n in param_optimizer if "pooler" not in n[0]]

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    def make_lr_scheduler(self, config, optimizer):
        lr_scheduler_type = getattr(config, "lr_scheduler_type", None)
        if lr_scheduler_type is None:
            return None

        lr_scheduler_config = getattr(config, lr_scheduler_type, {})
        if type(lr_scheduler_config) == NestedNamespace:
            lr_scheduler_config = vars(lr_scheduler_config)

        if "warmup" in lr_scheduler_type:
            lr_scheduler_config["num_training_steps"] = config.num_train_steps
            self.set_warmup_steps(lr_scheduler_config)

        lr_scheduler_config["optimizer"] = optimizer
        lr_scheduler = get_lr_schedulers()[lr_scheduler_type](**lr_scheduler_config)

        if lr_scheduler_type == "reduce_on_plateau":
            lr_scheduler = LearningRateWithMetricsWrapper(lr_scheduler)
        else:
            lr_scheduler = LearningRateWithoutMetricsWrapper(lr_scheduler)

        return lr_scheduler

    def set_warmup_steps(self, lr_scheduler_config):
        warmup_proportion = lr_scheduler_config.get("warmup_proportion", None)
        warmup_steps = lr_scheduler_config.get("warmup_steps", None)
        total_steps = lr_scheduler_config["num_training_steps"]

        if warmup_steps and warmup_proportion:
            raise ValueError("Check 'warmup_steps' and 'warmup_proportion'.")
        elif not warmup_steps and warmup_proportion:
            lr_scheduler_config["num_warmup_steps"] = int(total_steps * warmup_proportion) + 1
            del lr_scheduler_config["warmup_proportion"]
        elif warmup_steps and not warmup_proportion:
            # v2.11.0 change (argument name: warmup_steps -> num_warmup_steps)
            lr_scheduler_config["num_warmup_steps"] = warmup_steps
            del lr_scheduler_config["warmup_steps"]
        else:
            raise ValueError("Check 'warmup_steps' and 'warmup_proportion'.")


