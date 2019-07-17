# -*- coding: utf-8 -*-

import json
import logging
import os
import time
import random

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from claf import nsml
from claf.config.utils import pretty_json_dumps
from claf.learn.optimization.exponential_moving_avarage import EMA
from claf.learn.tensorboard import TensorBoard
from claf.learn import utils

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer
    Run experiment

    - train
    - train_and_evaluate
    - evaluate
    - evaluate_inference_latency
    - predict

    * Args:
        config: experiment overall config
        model: Model based on torch.nn.Module

    * Kwargs:
        log_dir: path to directory for save model and other options
        grad_max_norm: Clips gradient norm of an iterable of parameters.
        learning_rate_scheduler: PyTorch's Learning Rate Scheduler.
            (https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html)
        exponential_moving_average: the moving averages of all weights of the model are maintained
            with the exponential decay rate of {ema}.
        num_epochs: the number of maximun epochs (Default is 20)
        early_stopping_threshold: the number of early stopping threshold (Default is 10)
        max_eval_examples: print evaluation examples
        metric_key: metric score's control point
        verbose_step_count: print verbose step count (Default is 100)
        eval_and_save_step_count: evaluate valid_dataset then save every n step_count (Default is 'epoch')
    """

    def __init__(
        self,
        model,
        config={},
        log_dir="logs/experiment",
        grad_max_norm=None,
        gradient_accumulation_steps=1,
        learning_rate_scheduler=None,
        exponential_moving_average=None,
        num_epochs=20,
        early_stopping_threshold=10,
        max_eval_examples=5,
        metric_key=None,
        verbose_step_count=100,
        eval_and_save_step_count="epoch",
    ):
        assert metric_key is not None

        # CUDA
        self.use_multi_gpu = type(model) == torch.nn.DataParallel

        if getattr(model, "train_counter", None):
            self.train_counter = model.train_counter
        else:
            self.train_counter = utils.TrainCounter(display_unit=eval_and_save_step_count)

        self.model = model
        model_config = config.get("model", {})
        self.model_name = model_config.get("name", "model")
        self.set_model_base_properties(config, log_dir)

        # Logs
        os.makedirs(log_dir, exist_ok=True)
        self.tensorboard = TensorBoard(log_dir)
        self.metric_logs = {"best_epoch": 0, "best_global_step": 0, "best": None, "best_score": 0}
        self.training_logs = {"early_stopping_count": 0}

        # optimization options
        self.grad_max_norm = grad_max_norm

        if gradient_accumulation_steps is None:
            gradient_accumulation_steps = 1
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.learning_rate_scheduler = learning_rate_scheduler
        self.exponential_moving_average = exponential_moving_average
        if exponential_moving_average:
            self.exponential_moving_average = EMA(model, self.exponential_moving_average)

        # property
        self.num_epochs = num_epochs
        self.early_stopping = False
        self.early_stopping_threshold = early_stopping_threshold
        self.max_eval_examples = max_eval_examples
        self.metric_key = metric_key
        self.verbose_step_count = verbose_step_count
        self.eval_and_save_step_count = eval_and_save_step_count
        self.log_dir = log_dir

    def set_model_base_properties(self, config, log_dir):
        model = self.model
        if self.use_multi_gpu:
            model = self.model.module

        model.config = config
        model.log_dir = log_dir
        model.train_counter = self.train_counter
        assert model.is_ready() == True

    def train_and_evaluate(self, train_loader, valid_loader, optimizer):
        """ Train and Evaluate """
        start_time = time.time()

        for epoch in range(1, self.num_epochs + 1):
            self.train_counter.epoch = epoch

            # Training with metrics
            train_metrics = self._run_epoch(
                train_loader,
                valid_loader=valid_loader,
                is_training=True,
                optimizer=optimizer,
                verbose_step_count=self.verbose_step_count,
                eval_and_save_step_count=self.eval_and_save_step_count,
            )

            valid_metrics = None
            if self.eval_and_save_step_count == "epoch":
                valid_metrics = self._run_epoch(valid_loader, is_training=False)
                self._check_valid_results(valid_metrics, report=False)
                self.save(optimizer)

            self._report_metrics(train_metrics=train_metrics, valid_metrics=valid_metrics)
            self._estimate_remainig_time(start_time)

            if self.early_stopping:
                break

        self._report_trainings(start_time, train_loader=train_loader, valid_loader=valid_loader)

    def train(self, data_loader, optimizer):
        """ Train """
        start_time = time.time()

        for epoch in range(1, self.num_epochs + 1):
            self.train_counter.epoch = epoch

            metrics = self._run_epoch(
                data_loader,
                is_training=True,
                optimizer=optimizer,
                verbose_step_count=self.verbose_step_count,
            )

            self._report_metrics(train_metrics=metrics)
            self._estimate_remainig_time(start_time)
            self.save(optimizer)

        self._report_trainings(start_time, train_loader=data_loader)

    def evaluate(self, data_loader):
        """ Evaluate """
        eval_metrics = self._run_epoch(data_loader, is_training=False, disable_prograss_bar=False)

        self._report_metrics(tensorboard=False, valid_metrics=eval_metrics)

    def evaluate_inference_latency(self, raw_examples, raw_to_tensor_fn, token_key=None, max_latency=1000):
        """
        Evaluate with focusing inferece latency
        (Note: must use sorted synthetic data)

        * inference_latency: raw_data -> pre-processing -> model -> predict_value
                                (elapsed_time)               (elapsed_time)
        """

        logger.info("\n# Evaluate Inference Latency Mode.")
        self.model.eval()

        total_raw_to_tensor_time = 0
        tensor_to_predicts = []

        raw_example_items = tqdm(raw_examples.items())
        for _, raw_example in raw_example_items:
            # raw_data -> tensor
            raw_to_tensor_start_time = time.time()
            feature, helper = raw_to_tensor_fn(raw_example)
            raw_to_tensor_elapsted_time = time.time() - raw_to_tensor_start_time
            raw_to_tensor_elapsted_time *= 1000  # unit: sec -> ms

            total_raw_to_tensor_time += raw_to_tensor_elapsted_time

            # tensor to predict
            tensor_to_predict_start_time = time.time()
            output_dict = self.model(feature)
            tensor_to_predict_elapsed_time = time.time() - tensor_to_predict_start_time

            if "token_key" not in helper:
                raise ValueError(
                    "helper must have 'token_key' data for 1-example inference latency."
                )

            tensor_to_predict_elapsed_time *= 1000  # unit: sec -> ms
            tensor_to_predict = {
                "elapsed_time": tensor_to_predict_elapsed_time,
                "token_count": len(helper[helper["token_key"]]),
            }
            tensor_to_predicts.append(tensor_to_predict)

            if tensor_to_predict_elapsed_time > max_latency:
                raw_example_items.close()
                break

        total_tensor_to_predict = sum(
            [tensor_to_predict["elapsed_time"] for tensor_to_predict in tensor_to_predicts]
        )

        max_token_count_per_times = {}
        max_times = list(range(0, max_latency+1, 100))
        for t2p in sorted(tensor_to_predicts, key=lambda x: x["token_count"]):
            elapsed_time = t2p["elapsed_time"]
            token_count = t2p["token_count"]

            for max_time in max_times:
                if elapsed_time < max_time:
                    max_token_count_per_times[max_time] = token_count

        result = {
            "average_raw_to_tensor": total_raw_to_tensor_time / len(raw_examples),
            "average_tensor_to_predict": total_tensor_to_predict / len(raw_examples),
            "average_end_to_end": (total_raw_to_tensor_time + total_tensor_to_predict)
            / len(raw_examples),
            "tensor_to_predicts": tensor_to_predicts,
            "max_token_count_per_time": max_token_count_per_times
        }

        env = "gpu" if torch.cuda.is_available() else "cpu"
        file_name = f"{self.model_name}-{env}.json"
        with open(file_name, "w") as f:
            json.dump(result, f, indent=4)

        logger.info(f"saved inference_latency results. {file_name}")

    def _is_early_stopping(self, metrics):
        score = metrics[self.metric_key]

        if score > self.metric_logs["best_score"]:
            self.training_logs["early_stopping_count"] = 0
        else:
            self.training_logs["early_stopping_count"] += 1

        if self.training_logs["early_stopping_count"] >= self.early_stopping_threshold:
            self.training_logs["early_stopping"] = True
            return True
        else:
            return False

    def _report_metrics(self, tensorboard=True, train_metrics=None, valid_metrics=None):

        total_metrics = {}

        def update_metrics(metrics, category=""):
            if metrics is not None:
                for k, v in metrics.items():
                    total_metrics[f"{category}/{k}"] = v

        update_metrics(train_metrics, "train")
        update_metrics(valid_metrics, "valid")

        # TensorBoard
        if tensorboard:
            self.tensorboard.scalar_summaries(self.train_counter.get_display(), total_metrics)

        # Console
        metric_console = ""
        if train_metrics:
            metric_console += (
                f"\n# Epoch: [{self.train_counter.epoch}/{self.num_epochs}]: Metrics \n"
            )
        metric_console += json.dumps(total_metrics, indent=4)
        logger.info(metric_console)

        if valid_metrics:
            self._update_metric_logs(total_metrics)

    def _update_metric_logs(self, total_metrics):
        for k, v in total_metrics.items():
            if self.metric_logs.get(k, None) is None:
                self.metric_logs[k] = [v]
            else:
                self.metric_logs[k].append(v)

        valid_score = total_metrics.get(f"valid/{self.metric_key}", None)
        if valid_score and valid_score > self.metric_logs["best_score"]:
            logger.info(f" * Best validation score so far. ({self.metric_key}) : {valid_score}")
            self.metric_logs["best_score"] = valid_score
            self.metric_logs["best"] = total_metrics
            self.metric_logs["best_epoch"] = self.train_counter.epoch
            self.metric_logs["best_global_step"] = self.train_counter.global_step
        else:
            logger.info(
                f" * Current best validation score. ({self.metric_key}) : {self.metric_logs['best_score']}"
            )

    def _estimate_remainig_time(self, start_time):
        elapsed_time = time.time() - start_time
        estimated_time_remaining = elapsed_time * (
            (self.num_epochs - self.train_counter.epoch) / float(self.train_counter.epoch) - 1
        )
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(estimated_time_remaining))
        logger.info(f"Estimated training time remaining: {formatted_time} ")

    def _report_trainings(self, start_time, train_loader=None, valid_loader=None):
        elapsed_time = time.time() - start_time
        self.training_logs["elapsed_time"] = (time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),)

        if train_loader is not None:
            self.training_logs["train_dataset"] = json.loads(str(train_loader.dataset))
        if valid_loader is not None:
            self.training_logs["valid_dataset"] = json.loads(str(valid_loader.dataset))

    def _run_epoch(
        self,
        data_loader,
        valid_loader=None,
        is_training=True,
        optimizer=None,
        disable_prograss_bar=True,
        verbose_step_count=100,
        eval_and_save_step_count=None,
    ):
        """
        Run Epoch

        1. forward inputs to model
        2. (trainig) backprobagation
        3. update predictions
        4. make metrics
        """

        if is_training:
            logger.info("# Train Mode.")
            self.model.train()
        else:
            logger.info("# Evaluate Mode.")
            self.model.eval()

        # set dataset (train/valid)
        self._set_dataset_to_model(data_loader.dataset)

        metrics = {}
        predictions = {}

        epoch_loss = 0
        epoch_start_time = time.time()
        step_start_time = time.time()

        eval_example_count = 0

        for step, batch in enumerate(tqdm(data_loader, disable=disable_prograss_bar)):
            inputs = batch.to_dict()  # for DataParallel
            output_dict = self.model(**inputs)

            loss = output_dict["loss"]
            if self.use_multi_gpu:
                loss = loss.mean()
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            epoch_loss += loss.item()

            if is_training:
                # Training Verbose
                if self.train_counter.global_step == 0:
                    logger.info(f"  Start - Batch Loss: {loss.item():.5f}")

                if (
                    self.train_counter.global_step != 0
                    and self.train_counter.global_step % verbose_step_count == 0
                ):
                    step_elapsed_time = time.time() - step_start_time

                    logger.info(
                        f"  Step: {self.train_counter.global_step} Batch Loss: {loss.item():.5f}  {step_elapsed_time:.5f} sec"
                    )
                    self.tensorboard.scalar_summary(
                        self.train_counter.global_step, "train/batch_loss", loss.item()
                    )

                    step_start_time = time.time()

                loss.backward()

                if self.grad_max_norm:
                    clip_grad_norm_(self._get_model_parameters(), self.grad_max_norm)

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Backpropagation
                    if self.learning_rate_scheduler:
                        self.learning_rate_scheduler.step_batch(self.train_counter.global_step)

                    optimizer.step()
                    optimizer.zero_grad()
                    self.train_counter.global_step += 1

                    if self.exponential_moving_average:
                        for name, param in self.model.named_parameters():
                            if param.requires_grad:
                                param.data = self.exponential_moving_average(name, param.data)

                    # Evaluate then Save checkpoint
                    if (
                        valid_loader
                        and type(eval_and_save_step_count) == int
                        and self.train_counter.global_step % eval_and_save_step_count == 0
                    ):
                        with torch.no_grad():
                            valid_metrics = self._run_epoch(valid_loader, is_training=False)
                        self._check_valid_results(valid_metrics, report=True)
                        self.save(optimizer)

                        if is_training:  # roll-back to train mode
                            self.model.train()
                            self._set_dataset_to_model(data_loader.dataset)
            else:
                if eval_example_count < self.max_eval_examples:
                    total_step_count = int(len(data_loader) / data_loader.batch_size)
                    random_num = random.randint(0, total_step_count)

                    if random_num <= self.max_eval_examples:
                        eval_example_predictions = {}
                        self._update_predictions(eval_example_predictions, output_dict)

                        random_index = random.randint(0, data_loader.batch_size)
                        self._print_examples(random_index, inputs, eval_example_predictions)
                        eval_example_count += 1

            self._update_predictions(predictions, output_dict)

        epoch_loss /= len(data_loader)
        epoch_elapsed_time = time.time() - epoch_start_time

        logger.info("Epoch duration: " + time.strftime("%H:%M:%S", time.gmtime(epoch_elapsed_time)))

        # Updat metrics
        metrics["loss"] = epoch_loss
        metrics["epoch_time"] = epoch_elapsed_time
        metrics.update(self._make_metrics(predictions))  # model metric

        return metrics

    def _set_dataset_to_model(self, dataset):
        if self.use_multi_gpu:
            self.model.module.dataset = dataset
        else:
            self.model.dataset = dataset

    def _get_model_parameters(self):
        if self.use_multi_gpu:
            return self.model.module.parameters()
        else:
            return self.model.parameters()

    def _check_valid_results(self, metrics, report=False):
        if self.learning_rate_scheduler:
            # The LRScheduler API is agnostic to whether your schedule requires a validation metric -
            # if it doesn't, the validation metric passed here is ignored.
            this_epoch_val_metric = metrics[self.metric_key]
            self.learning_rate_scheduler.step(this_epoch_val_metric, self.train_counter.global_step)

        if self._is_early_stopping(metrics):
            self.early_stopping = True
            logger.info(" --- Early Stopping. --- ")

        if report:
            self._report_metrics(valid_metrics=metrics)

    def _make_metrics(self, predictions):
        model = self.model
        if self.use_multi_gpu:
            model = model.module

        model.train_counter = self.train_counter
        return model.make_metrics(predictions)

    def _update_predictions(self, predictions, output_dict):
        if self.use_multi_gpu:
            predictions.update(self.model.module.make_predictions(output_dict))
        else:
            predictions.update(self.model.make_predictions(output_dict))

    def _print_examples(self, index, inputs, predictions):
        try:
            if self.use_multi_gpu:
                self.model.module.print_examples(index, inputs, predictions)
            else:
                self.model.print_examples(index, inputs, predictions)
        except IndexError:
            pass

    def predict(self, raw_feature, raw_to_tensor_fn, arguments, interactive=False):
        """ Inference / Predict """

        self.model.eval()
        with torch.no_grad():
            if interactive:  # pragma: no cover
                while True:
                    for k in raw_feature:
                        raw_feature[k] = utils.get_user_input(k)

                    tensor_feature, helper = raw_to_tensor_fn(raw_feature)
                    output_dict = self.model(tensor_feature)

                    arguments.update(raw_feature)
                    predict = self.model.predict(output_dict, arguments, helper)
                    print(f"Predict: {pretty_json_dumps(predict)} \n")
            else:
                tensor_feature, helper = raw_to_tensor_fn(raw_feature)
                output_dict = self.model(tensor_feature)

                return self.model.predict(output_dict, arguments, helper)

    def save(self, optimizer):
        # set all config to model
        model = self.model
        if self.use_multi_gpu:
            model = self.model.module

        model.train_counter = self.train_counter
        model.metrics = self.metric_logs

        if nsml.IS_ON_NSML:
            nsml.save(self.train_counter.get_display())
        else:
            utils.save_checkpoint(self.log_dir, model, optimizer)
