# -*- coding: utf-8 -*-


from claf.config import args
from claf.learn.experiment import Experiment
from claf.learn.mode import Mode


if __name__ == "__main__":
    experiment = Experiment(Mode.PREDICT, args.config(mode=Mode.PREDICT))
    result = experiment()

    print(f"Predict: {result}")
