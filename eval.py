# -*- coding: utf-8 -*-


from claf.config import args
from claf.learn.experiment import Experiment
from claf.learn.mode import Mode


if __name__ == "__main__":
    config = args.config(mode=Mode.EVAL)

    mode = Mode.EVAL
    if config.inference_latency: # evaluate inference_latency
        mode = Mode.INFER_EVAL

    experiment = Experiment(mode, config)
    experiment()
