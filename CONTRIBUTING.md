# Contributing to CLAF

First of all, thank you for considering contributing to CLAF. It's people like you that make CLaF such a great framework.

Following these guidelines helps to communicate that you respect the time of the developers managing and developing this open source project. In return, they should reciprocate that respect in addressing your issue, assessing changes, and helping you finalize your pull requests.

If you are not familiar with creating a Pull Request, here are some guides:
    
- [Create a Pull Request](https://help.github.com/articles/creating-a-pull-request/)

## Bug & Simple features

1. Search on [Issues](https://github.com/naver/claf/issues)
2. If there are similar issues, add Comments to those issues, otherwise, create new ones

3. Check `pytest`, `black (lint)` before Pull Request
    - [pytest](https://github.com/pytest-dev/pytest)
        - Add unittest to the `tests/claf` folder and integration test code if necessary
        - Run test with coverage ```pytest --cov-config .coveragerc --cov-report html:cov_html --cov=rqa tests``` 
    - [Black](https://github.com/ambv/black) (lint)
        - ```black claf  -l 120 ``` (reformat your code)
4. Clean up your work to create a Pull Request

(* When adding a new function (model, optimizer and so on), add it to `claf.config.ars` with a description.)

e.g. Exponential Learning Rate Scheduler
 
```
 # ExponentialLR:
  --exponential.gamma OPTIMIZER.EXPONENTIAL.GAMMA
                            Multiplicative factor of learning rate decay.
                            Default: 0.1.
  --exponential.last_epoch OPTIMIZER.EXPONENTIAL.LAST_EPOCH
                            The index of last epoch.
                            Default: -1.
```

## The structure of the framework

1. Post it on the issue and discuss it with maintainers.
2. After discuss, organize according to priority and start working on.
