# -*- coding: utf-8 -*-

from claf.config import args
from claf.learn.experiment import Experiment
from claf.learn.mode import Mode


if __name__ == "__main__":
    experiment = Experiment(Mode.TRAIN, args.config(mode=Mode.TRAIN))
    experiment()
