
import atexit
import logging
from pathlib import Path

import torch

from claf import nsml
from claf.config.factory import (
    DataReaderFactory,
    DataLoaderFactory,
    TokenMakersFactory,
    ModelFactory,
    OptimizerFactory,
)
from claf import utils as common_utils
from claf.config.args import NestedNamespace
from claf.config.utils import convert_config2dict, pretty_json_dumps, set_global_seed
from claf.tokens.text_handler import TextHandler
from claf.learn.mode import Mode
from claf.learn.trainer import Trainer
from claf.learn import utils


logger = logging.getLogger(__name__)


class Experiment:
    """
    Experiment settings with config.

    * Args:
        mode: Mode (ex. TRAIN, EVAL, INFER_EVAL, PREDICT)
        config: (NestedNamespace) Argument config according to mode
    """

    def __init__(self, mode, config):
        common_utils.set_logging_config(mode, config)

        self.argument = (
            config
        )  # self.config (experiment overall config) / config (argument according to mode)
        self.config = config
        self.mode = mode

        self.common_setting(mode, config)
        if mode != Mode.TRAIN:  # evaluate and predict
            self.load_setting()

            # Set evaluation config
            if mode.endswith(Mode.EVAL):
                self.config.data_reader.train_file_path = ""
                self.config.data_reader.valid_file_path = self.argument.data_file_path
                self.config.cuda_devices = self.argument.cuda_devices
                self.config.iterator.cuda_devices = self.argument.cuda_devices

                if getattr(self.argument, "inference_latency", None):
                    self.config.max_latency = self.argument.inference_latency

        self.predict_settings = None

    def common_setting(self, mode, config):
        """ Common Setting - experiment config, use_gpu and cuda_device_ids """
        self.config_dict = convert_config2dict(config)

        cuda_devices = self._get_cuda_devices()
        self.config.cuda_devices = cuda_devices
        self.config.slack_url = getattr(self.config, "slack_url", False)

    def _get_cuda_devices(self):
        if getattr(self.config, "use_gpu", None) is None:
            self.config.use_gpu = torch.cuda.is_available() or nsml.IS_ON_NSML

        if self.config.use_gpu:
            if nsml.IS_ON_NSML:
                return list(range(self.config.gpu_num))
            else:
                return self.config.cuda_devices
        else:
            return None

    def load_setting(self):
        """ Load Setting - need to load checkpoint case (ex. evaluate and predict) """
        cuda_devices = self.argument.cuda_devices
        checkpoint_path = self.argument.checkpoint_path
        prev_cuda_device_id = getattr(self.argument, "prev_cuda_device_id", None)

        self.model_checkpoint = self._read_checkpoint(
            cuda_devices, checkpoint_path, prev_cuda_device_id=prev_cuda_device_id
        )
        self._set_saved_config(cuda_devices)

    def _read_checkpoint(self, cuda_devices, checkpoint_path, prev_cuda_device_id=None):
        if cuda_devices == "cpu":
            return torch.load(checkpoint_path, map_location="cpu")  # use CPU

        if torch.cuda.is_available():
            checkpoint = torch.load(
                checkpoint_path,
                map_location={
                    f"cuda:{prev_cuda_device_id}": f"cuda:{cuda_devices[0]}"
                },  # different cuda_device id case (save/load)
            )
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")  # use CPU
        return checkpoint

    def _set_saved_config(self, cuda_devices):
        saved_config_dict = self.model_checkpoint["config"]
        self.config_dict = saved_config_dict

        logger.info("Load saved_config ...")
        logger.info(pretty_json_dumps(saved_config_dict))

        saved_config = NestedNamespace()
        saved_config.load_from_json(saved_config_dict)

        is_use_gpu = self.config.use_gpu

        self.config = saved_config
        self.config.use_gpu = is_use_gpu
        self.config.cuda_devices = cuda_devices

    def __call__(self):
        """ Run Trainer """

        set_global_seed(self.config.seed_num)  # For Reproducible

        if self.mode == Mode.TRAIN:
            # exit trigger slack notification
            if self.config.slack_url:
                atexit.register(utils.send_message_to_slack)

            train_loader, valid_loader, optimizer = self.set_train_mode()

            assert train_loader is not None
            assert optimizer is not None

            if valid_loader is None:
                self.trainer.train(train_loader, optimizer)
            else:
                self.trainer.train_and_evaluate(train_loader, valid_loader, optimizer)
            self._summary_experiments()

        elif self.mode == Mode.EVAL:
            valid_loader = self.set_eval_mode()

            assert valid_loader is not None
            return self.trainer.evaluate(valid_loader)

        elif self.mode == Mode.INFER_EVAL:
            raw_examples, raw_to_tensor_fn = self.set_eval_inference_latency_mode()

            assert raw_examples is not None
            assert raw_to_tensor_fn is not None
            return self.trainer.evaluate_inference_latency(raw_examples, raw_to_tensor_fn, max_latency=self.config.max_latency)

        elif self.mode.endswith(Mode.PREDICT):
            raw_features, raw_to_tensor_fn, arguments = self.set_predict_mode()

            assert raw_features is not None
            assert raw_to_tensor_fn is not None
            return self.trainer.predict(
                raw_features,
                raw_to_tensor_fn,
                arguments,
                interactive=arguments.get("interactive", False),
            )
        else:
            raise ValueError(f"unknown mode: {self.mode}")

    def set_train_mode(self):
        """
        Training Mode

        - Pipeline
          1. read raw_data (DataReader)
          2. build vocabs (DataReader, Token)
          3. indexing tokens (DataReader, Token)
          4. convert to DataSet (DataReader)
          5. create DataLoader (DataLoader)
          6. define model and optimizer
          7. run!
        """
        logger.info("Config. \n" + pretty_json_dumps(self.config_dict) + "\n")

        data_reader, token_makers = self._create_data_and_token_makers()
        datas, helpers = data_reader.read()

        # Token & Vocab
        text_handler = TextHandler(token_makers, lazy_indexing=True)
        texts = data_reader.filter_texts(datas)

        token_counters = text_handler.make_token_counters(texts, config=self.config)
        text_handler.build_vocabs(token_counters)
        text_handler.index(datas, data_reader.text_columns)

        # iterator
        datasets = data_reader.convert_to_dataset(datas, helpers=helpers)  # with name

        self.config.iterator.cuda_devices = self.config.cuda_devices
        train_loader, valid_loader, test_loader = self._create_by_factory(
            DataLoaderFactory, self.config.iterator, param={"datasets": datasets}
        )

        # calculate 'num_train_steps'
        num_train_steps = self._get_num_train_steps(train_loader)
        self.config.optimizer.num_train_steps = num_train_steps

        checkpoint_dir = Path(self.config.trainer.log_dir) / "checkpoint"
        checkpoints = None
        if checkpoint_dir.exists():
            checkpoints = self._load_exist_checkpoints(checkpoint_dir)  # contain model and optimizer

        if checkpoints is None:
            model = self._create_model(token_makers, helpers=helpers)
            op_dict = self._create_by_factory(
                OptimizerFactory, self.config.optimizer, param={"model": model}
            )
        else:
            model = self._create_model(token_makers, checkpoint=checkpoints)
            op_dict = self._create_by_factory(
                OptimizerFactory, self.config.optimizer, param={"model": model}
            )
            utils.load_optimizer_checkpoint(op_dict["optimizer"], checkpoints)

        self.set_trainer(model, op_dict=op_dict)
        return train_loader, valid_loader, op_dict["optimizer"]

    def _create_data_and_token_makers(self):
        token_makers = self._create_by_factory(TokenMakersFactory, self.config.token)
        tokenizers = token_makers["tokenizers"]
        del token_makers["tokenizers"]

        self.config.data_reader.tokenizers = tokenizers
        data_reader = self._create_by_factory(DataReaderFactory, self.config.data_reader)
        return data_reader, token_makers

    def _create_by_factory(self, factory, item_config, param={}):
        return factory(item_config).create(**param)

    def _get_num_train_steps(self, train_loader):
        train_set_size = len(train_loader.dataset)
        batch_size = self.config.iterator.batch_size
        gradient_accumulation_steps = getattr(self.config.optimizer, "gradient_accumulation_steps", 1)
        num_epochs = self.config.trainer.num_epochs

        one_epoch_steps = int(train_set_size / batch_size / gradient_accumulation_steps)
        if one_epoch_steps == 0:
            one_epoch_steps = 1
        num_train_steps = one_epoch_steps * num_epochs
        return num_train_steps

    def _load_exist_checkpoints(self, checkpoint_dir):  # pragma: no cover
        checkpoints = utils.get_sorted_path(checkpoint_dir, both_exist=True)

        train_counts = list(checkpoints.keys())
        if not train_counts:
            return None

        seperator = "-" * 50
        message = f"{seperator}\n !! Find exist checkpoints {train_counts}.\n If you want to recover, input train_count in list.\n If you don't want to recover, input 0.\n{seperator}"
        selected_train_count = common_utils.get_user_input(message)

        if selected_train_count == 0:
            return None

        model_path = checkpoints[selected_train_count]["model"]
        model_checkpoint = self._read_checkpoint(self.config.cuda_devices, model_path)

        optimizer_path = checkpoints[selected_train_count]["optimizer"]
        optimizer_checkpoint = self._read_checkpoint("cpu", optimizer_path)

        checkpoints = {}
        checkpoints.update(model_checkpoint)
        checkpoints.update(optimizer_checkpoint)
        return checkpoints

    def _create_model(self, token_makers, checkpoint=None, helpers=None):
        if checkpoint is None:
            assert helpers is not None
            first_key = next(iter(helpers))
            helper = helpers[first_key]  # get first helper
            model_init_params = helper.get("model", {})
            predict_helper = helper.get("predict_helper", {})
        else:
            model_init_params = checkpoint.get("init_params", {})
            predict_helper = checkpoint.get("predict_helper", {})

        model_params = {"token_makers": token_makers}
        model_params.update(model_init_params)

        model = self._create_by_factory(
            ModelFactory, self.config.model, param=model_params
        )
        # Save params
        model.init_params = model_init_params
        model.predict_helper = predict_helper

        if checkpoint is not None:
            model = utils.load_model_checkpoint(model, checkpoint)
        model = self._set_gpu_env(model)
        return model

    def _set_gpu_env(self, model):
        if self.config.use_gpu:
            cuda_devices = self._get_cuda_devices()
            num_gpu = len(cuda_devices)

            use_multi_gpu = num_gpu > 1
            if use_multi_gpu:
                model = torch.nn.DataParallel(model, device_ids=cuda_devices)
            model.cuda()
        else:
            num_gpu = 0

        num_gpu_state = num_gpu
        if num_gpu > 1:
            num_gpu_state += " (Multi-GPU)"
        logger.info(f"use_gpu: {self.config.use_gpu} num_gpu: {num_gpu_state}, distributed training: False, 16-bits trainiing: False")
        return model

    def set_trainer(self, model, op_dict={}, save_params={}):
        trainer_config = vars(self.config.trainer)
        trainer_config["config"] = self.config_dict
        trainer_config["model"] = model
        trainer_config["learning_rate_scheduler"] = op_dict.get("learning_rate_scheduler", None)
        trainer_config["exponential_moving_average"] = op_dict.get(
            "exponential_moving_average", None
        )
        self.trainer = Trainer(**trainer_config)

        # Set NSML
        if nsml.IS_ON_NSML:
            utils.bind_nsml(model, optimizer=op_dict.get("optimizer", None))
            if getattr(self.config.nsml, "pause", None):
                nsml.paused(scope=locals())

    def _summary_experiments(self):
        hr_text = "-" * 50
        summary_logs = f"\n\n\nExperiment Summary. {nsml.SESSION_NAME}\n{hr_text}\n"
        summary_logs += f"Config.\n{pretty_json_dumps(self.config_dict)}\n{hr_text}\n"
        summary_logs += (
            f"Training Logs.\n{pretty_json_dumps(self.trainer.training_logs)}\n{hr_text}\n"
        )
        summary_logs += f"Metric Logs.\n{pretty_json_dumps(self.trainer.metric_logs)}"

        logger.info(summary_logs)

        if self.config.slack_url:  # pragma: no cover
            simple_summary_title = f"Session Name: {nsml.SESSION_NAME} "
            if getattr(self.config, "base_config", None):
                simple_summary_title += f"({self.config.base_config})"

            simple_summary_logs = f" - Dataset: {self.config.data_reader.dataset} \n"
            simple_summary_logs += f" - Model: {self.config.model.name}"

            best_metrics = {"epoch": self.trainer.metric_logs["best_epoch"]}
            best_metrics.update(self.trainer.metric_logs["best"])

            simple_summary_logs += f" - Best metrics.\n {pretty_json_dumps(best_metrics)} "

            utils.send_message_to_slack(self.config.slack_url, title=simple_summary_title, message=simple_summary_logs)

    def set_eval_mode(self):
        """
        Evaluate Mode

        - Pipeline
          1. read raw_data (DataReader)
          2. load vocabs from checkpoint (DataReader, Token)
          3. indexing tokens (DataReader, Token)
          4. convert to DataSet (DataReader)
          5. create DataLoader (DataLoader)
          6. define and load model
          7. run!
        """

        data_reader, token_makers = self._create_data_and_token_makers()

        # DataReader
        datas, helpers = data_reader.read()

        # Token & Vocab
        vocabs = utils.load_vocabs(self.model_checkpoint)
        for token_name, token_maker in token_makers.items():
            token_maker.set_vocab(vocabs[token_name])

        text_handler = TextHandler(token_makers, lazy_indexing=False)
        text_handler.index(datas, data_reader.text_columns)

        # iterator
        datasets = data_reader.convert_to_dataset(datas, helpers=helpers)  # with name

        self.config.iterator.cuda_devices = self.config.cuda_devices
        _, valid_loader, _ = self._create_by_factory(
            DataLoaderFactory, self.config.iterator, param={"datasets": datasets}
        )

        # Model
        model = self._create_model(token_makers, checkpoint=self.model_checkpoint)
        self.set_trainer(model)

        return valid_loader

    def set_eval_inference_latency_mode(self):
        """
        Evaluate Inference Latency Mode

        - Pipeline
          1. read raw_data (DataReader)
          2. load vocabs from checkpoint (DataReader, Token)
          3. define raw_to_tensor_fn (DataReader, Token)
          4. define and load model
          5. run!
        """
        data_reader, token_makers = self._create_data_and_token_makers()

        # Token & Vocab
        vocabs = utils.load_vocabs(self.model_checkpoint)
        for token_name, token_maker in token_makers.items():
            token_maker.set_vocab(vocabs[token_name])

        text_handler = TextHandler(token_makers, lazy_indexing=False)

        _, helpers = data_reader.read()
        raw_examples = helpers["valid"]["examples"]

        cuda_device = self.config.cuda_devices[0] if self.config.use_gpu else None
        raw_to_tensor_fn = text_handler.raw_to_tensor_fn(data_reader, cuda_device=cuda_device)

        # Model
        model = self._create_model(token_makers, checkpoint=self.model_checkpoint)
        self.set_trainer(model)

        return raw_examples, raw_to_tensor_fn

    def predict(self, raw_features):
        if self.predict_settings is None:
            raise ValueError(
                "To use 'predict()', you must call 'set_predict_mode()' first, with preload=True parameter"
            )

        raw_to_tensor_fn = self.predict_settings["raw_to_tensor_fn"]
        arguments = self.predict_settings["arguments"]
        arguments.update(raw_features)

        assert raw_features is not None
        assert raw_to_tensor_fn is not None
        return self.trainer.predict(
            raw_features,
            raw_to_tensor_fn,
            arguments,
            interactive=arguments.get("interactive", False),
        )

    def set_predict_mode(self, preload=False):
        """
        Predict Mode

        - Pipeline
          1. read raw_data (Argument)
          2. load vocabs from checkpoint (DataReader, Token)
          3. define raw_to_tensor_fn (DataReader, Token)
          4. define and load model
          5. run!
        """

        data_reader, token_makers = self._create_data_and_token_makers()

        # Token & Vocab
        vocabs = utils.load_vocabs(self.model_checkpoint)
        for token_name, token_maker in token_makers.items():
            token_maker.set_vocab(vocabs[token_name])

        text_handler = TextHandler(token_makers, lazy_indexing=False)

        # Set predict config
        if self.argument.interactive:
            raw_features = {feature_name: "" for feature_name in data_reader.text_columns}
        else:
            raw_features = {}
            for feature_name in data_reader.text_columns:
                feature = getattr(self.argument, feature_name, None)
                # if feature is None:
                # raise ValueError(f"--{feature_name} argument is required!")
                raw_features[feature_name] = feature

        cuda_device = self.config.cuda_devices[0] if self.config.use_gpu else None
        raw_to_tensor_fn = text_handler.raw_to_tensor_fn(
            data_reader,
            cuda_device=cuda_device,
            helper=self.model_checkpoint.get("predict_helper", {})
        )

        # Model
        model = self._create_model(token_makers, checkpoint=self.model_checkpoint)
        self.set_trainer(model)

        arguments = vars(self.argument)

        if preload:
            self.predict_settings = {"raw_to_tensor_fn": raw_to_tensor_fn, "arguments": arguments}
        else:
            return raw_features, raw_to_tensor_fn, arguments
