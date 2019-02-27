"""
    https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers.py
"""

import math
from overrides import overrides
import torch


def get_lr_schedulers():
    return {
        "step": torch.optim.lr_scheduler.StepLR,
        "multi_step": torch.optim.lr_scheduler.MultiStepLR,
        "exponential": torch.optim.lr_scheduler.ExponentialLR,
        "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
        "warmup": WarmUpLR,
        "noam": NoamLR,
    }


class LearningRateScheduler:
    def __init__(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def step(self, metric, epoch=None):
        raise NotImplementedError

    def step_batch(self, batch_num_total):
        if batch_num_total is not None:
            if hasattr(self.lr_scheduler, "step_batch"):
                self.lr_scheduler.step_batch(batch_num_total)
            return


class LearningRateWithoutMetricsWrapper(LearningRateScheduler):
    """
    A wrapper around learning rate schedulers that do not require metrics
    """

    def __init__(
        self, lr_scheduler: torch.optim.lr_scheduler._LRScheduler
    ) -> None:  # pylint: disable=protected-access
        super().__init__(lr_scheduler)
        self.lr_scheduler = lr_scheduler

    @overrides
    def step(self, metric, epoch=None):
        self.lr_scheduler.step(epoch)


class LearningRateWithMetricsWrapper(LearningRateScheduler):
    """
    A wrapper around learning rate schedulers that require metrics,
    At the moment there is only a single instance of this lrs. It is the ReduceLROnPlateau
    """

    def __init__(self, lr_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau) -> None:
        super().__init__(lr_scheduler)
        self.lr_scheduler = lr_scheduler

    @overrides
    def step(self, metric, epoch=None):
        if metric is None:
            raise ValueError(
                "The reduce_on_plateau learning rate scheduler requires "
                "a validation metric to compute the schedule and therefore "
                "must be used with a validation dataset."
            )
        self.lr_scheduler.step(metric, epoch)


class WarmUpLR(torch.optim.lr_scheduler._LRScheduler):
    """
        Learning Rate Scheduler

        a learning rate warm-up scheme with an inverse exponential increase
         from 0.0 to {learning_rate} in the first {final_step}.

        * Args:
            optimizer: torch.optim.Optimizer
            final_step: The number of steps to exponential increase the learning rate.
    """

    def __init__(self, optimizer, final_step=1000, last_epoch=-1):
        super().__init__(optimizer, last_epoch=last_epoch)
        self.final_step = final_step

        learning_rate = optimizer.param_groups[0]["lr"]
        self.crit = learning_rate / math.log2(final_step)

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            break
        return param_group["lr"]

    def step_batch(self, num_step):
        if num_step <= self.final_step:
            self.adjust_lr(num_step)

    def step(self, epoch=None):
        pass

    def adjust_lr(self, step):
        lr = self.crit * math.log2(step + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


class NoamLR(torch.optim.lr_scheduler._LRScheduler):  # pylint: disable=protected-access
    """
    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    Parameters
    ----------
    model_size : ``int``, required.
        The hidden size parameter which dominates the number of parameters in your model.
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.
    factor : ``float``, optional (default = 1.0).
        The overall scale factor for the learning rate decay.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model_size: int,
        warmup_steps: int,
        factor: float = 1.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.model_size = model_size
        super().__init__(optimizer, last_epoch=last_epoch)

    def step(self, epoch=None):
        pass

    def step_batch(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, learning_rate in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = learning_rate

    def get_lr(self):
        step = max(self.last_epoch, 1)
        scale = self.factor * (
            self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )

        return [scale for _ in range(len(self.base_lrs))]
