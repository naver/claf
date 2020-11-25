<p align="center">
    <img src="images/logo.png" style="inline" width=300>
</p>

<h4 align="center">Clova Language Framework</h4>

<p align="center">
    <a href="https://naver.github.io/claf">
        <img src="https://img.shields.io/badge/docs-passing-brightgreen.svg" alt="Documentation Status">
    </a>
    <a href="https://travis-ci.org/naver/claf">
        <img src='https://travis-ci.org/naver/claf.svg?branch=master'/>
    </a>
    <a href="https://github.com/ambv/black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black">
    <a href="https://codecov.io/gh/naver/claf">
    <img src="https://codecov.io/gh/naver/claf/branch/master/graph/badge.svg" />
  </a>
</p>

---

# CLaF: Clova Language Framework


- [Full Documentation](https://naver.github.io/claf/)
- [Dataset And Model](https://naver.github.io/claf/docs/_build/html/contents/dataset_and_model.html)
- [Pretrained Vector](https://naver.github.io/claf/docs/_build/html/contents/pretrained_vector.html)
- [Tokens](https://naver.github.io/claf/docs/_build/html/contents/tokens.html): `Tokenizers` and `TokenMakers`
- List of [BaseConfig](#baseconfig)

| Task | Language | Dataset | Model |
| ---- | -------- | ------- | ----- |
| Multi-Task Learning | English | [GLUE Benchmark](https://gluebenchmark.com/), [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) | `MT-DNN (BERT)` |
| Natural Language Understanding | English | [GLUE Benchmark](https://gluebenchmark.com/) | `BERT`, `RoBERTa` |
| Named Entity Recognition | English | CoNLL 2003 | `BERT` |
| Question Answering | Korean | [KorQuAD v1.0](https://korquad.github.io/category/1.0_KOR.html) | `BiDAF`, `DocQA`, `BERT` |
| Question Answering | Engilsh | [SQuAD v1.1 and v2.0](https://rajpurkar.github.io/SQuAD-explorer/) | - v1.1: `BiDAF`, `DrQA`, `DocQA`, `DocQA+ELMo`, `QANet`, `BERT`, `RoBERTa` <br/> - v2.0: `BiDAF + No Answer`, `DocQA + No Answer` |
| Semantic Parsing | English | [WikiSQL](https://github.com/salesforce/WikiSQL) | `SQLNet` |


- Reports
    - [GLUE](https://naver.github.io/claf/docs/_build/html/reports/glue.html)
    - [KorQuAD](https://naver.github.io/claf/docs/_build/html/reports/korquad.html)
    - [SQuAD](https://naver.github.io/claf/docs/_build/html/reports/squad.html)
    - [WikiSQL](https://naver.github.io/claf/docs/_build/html/reports/wikisql.html)
- Summary (1-example Inference Latency)
    - [Reading Comprehension](https://naver.github.io/claf/docs/_build/html/summary/reading_comprehension.html)


- List of [MachineConfig](#machine)

| Name | Language | Pipeline | Note |
| ---- | -------- | ------- | ----- |
| KoWiki | Korean | `Wiki Dumps` -> `Document Retrieval` -> `Reading Comprehension` | - |
| NLU | All | `Query` -> `Intent Classification` & `Token Classification (Slot)` -> `Template Matching` | - |

---


## Table of Contents
- [Overview](#overview)
    - [Features](#features)
- [Installation](#installation) 
    - [Requirements](#requirements)
    - [Install via pip](#install-via-pip)
- [Experiment](#experiment)
	- [Usage](#usage)
	    - [Training](#training) 
	    - [Evaluate](#evaluate) 
	    - [Predict](#predict) 
	    - [Docker Images](#docker-images)
- [Machine](#machine)
- [Contributing](#contributing)
- [Maintainers](#maintainers)
- [Citing](#citing)
- [License](#license)


---


## Overview

**CLaF** is a Language Framework built on PyTorch that provides following two high-level features:

- `Experiment` enables the control of training flow in general NLP by offering various `TokenMaker` methods. 
    - CLaF is inspired by the design principle of [AllenNLP](https://github.com/allenai/allennlp) such as the higher level concepts and reusable code, but mostly based on PyTorch’s common module, so that user can easily modify the code on their demands.  
- `Machine` helps to combine various modules to build a NLP Machine in one place.
    - There are knowledge-based, components and trained experiments which infer 1-example in modules.

### Features

- **Multilingual** modeling support (currently, English and Korean are supported).
- Light weighted **Systemization** and Modularization.
- Easy extension and implementation of models.
- A wide variation of **Experiments** with reproducible and comprehensive logging
- The metrics for services such as "1\-example inference latency" are provided.
- Easy to build of a NLP **Machine** by combining modules.


## Installation

### Requirements

- Python 3.6
- PyTorch >= 1.3.1
- [MeCab](https://bitbucket.org/eunjeon/mecab-ko) for Korean Tokenizer
    - ```sh script/install_mecab.sh```

It is recommended to use the virtual environment.  
[Conda](https://conda.io/docs/download.html) is the easiest way to set up a virtual environment.

```
conda create -n claf python=3.6
conda activate claf

(claf) ✗ pip install -r requirements.txt
```

### Install via pip

Commands to install via pip 

```
pip install claf
```


## Experiment

- Training Flow

![images](images/claf-experiment.001.png)


### Usage

#### Training

![images](images/training_config_mapping.png)


1. only Arguments

	```
	python train.py --train_file_path {file_path} --valid_file_path {file_path} --model_name {name} ...
	```

2. only BaseConfig (skip `/base_config` path)

	```
	python train.py --base_config {base_config}
	```
	
3. BaseConfig + Arguments

	```
	python train.py --base_config {base_config} --learning_rate 0.002
	```
	
	- Load BaseConfig then overwrite `learning_rate` to 0.002


#### BaseConfig

Declarative experiment config (.json, .ymal)

- Simply matching with object's parameters
- Exists samples in `/base_config` directory

##### Defined BaseConfig

```
Base Config:
  --base_config BASE_CONFIG
    Use pre-defined base_config:
    []


    * CoNLL 2003:
    ['conll2003/bert_large_cased']

    * GLUE:
    ['glue/qqp_roberta_base', 'glue/qnli_bert_base', 'glue/rte_bert_base', 'glue/wnli_roberta_base', 'glue/mnlim_roberta_base', 'glue/wnli_bert_base', 'glue/mnlimm_roberta_base', 'glue/cola_bert_base', 'glue/mrpc_bert_base', 'glue/mnlimm_bert_base', 'glue/stsb_bert_base', 'glue/mnlim_bert_base', 'glue/qqp_bert_base', 'glue/rte_roberta_base', 'glue/qnli_roberta_base', 'glue/sst_bert_base', 'glue/mrpc_roberta_base', 'glue/cola_roberta_base', 'glue/stsb_roberta_base', 'glue/sst_roberta_base']

    * KorQuAD:
    ['korquad/bert_base_multilingual_cased', 'korquad/bidaf', 'korquad/bert_base_multilingual_uncased', 'korquad/docqa']

    * SQuAD:
    ['squad/bert_large_uncased', 'squad/bidaf', 'squad/drqa_paper', 'squad/drqa', 'squad/bert_base_uncased', 'squad/qanet', 'squad/docqa+elmo', 'squad/bidaf_no_answer', 'squad/docqa_no_answer', 'squad/qanet_paper', 'squad/bidaf+elmo', 'squad/docqa']

    * WikiSQL:
    ['wikisql/sqlnet']
```


#### Evaluate

```
python eval.py <data_path> <model_checkpoint_path>
```

- Example

```
✗ python eval.py data/squad/dev-v1.1.json logs/squad/bidaf/checkpoint/model_19.pkl
...
[INFO] - {
    "valid/loss": 2.59111491665019,
    "valid/epoch_time": 60.7434446811676,
    "valid/start_acc": 63.17880794701987,
    "valid/end_acc": 67.19016083254493,
    "valid/span_acc": 54.45600756859035,
    "valid/em": 68.10785241248817,
    "valid/f1": 77.77963381714842
}
# write predictions files (<log_dir>/predictions/predictions-valid-19.json)
```

- 1-example Inference Latency ([Summary](docs/_build/html/reports/summary.html))

```
✗ python eval.py data/squad/dev-v1.1.json logs/squad/bidaf/checkpoint/model_19.pkl
...
# Evaluate Inference Latency Mode.
...
[INFO] - saved inference_latency results. bidaf-cpu.json  # file_format: {model_name}-{env}.json
```

#### Predict

```
python predict.py <model_checkpoint_path> --<arguments>
```

- Example

```
✗ python predict.py logs/squad/bidaf/checkpoint/model_19.pkl \
    --question "When was the last Super Bowl in California?" \
    --context "On May 21, 2013, NFL owners at their spring meetings in Boston voted and awarded the game to Levi's Stadium. The $1.2 billion stadium opened in 2014. It is the first Super Bowl held in the San Francisco Bay Area since Super Bowl XIX in 1985, and the first in California since Super Bowl XXXVII took place in San Diego in 2003."

>>> Predict: {'text': '2003', 'score': 4.1640071868896484}
```

#### Docker Images

- [Docker Hub](https://hub.docker.com/u/claf)
- Run with Docker Image
    - Pull docker image
        ```✗ docker pull claf/claf:latest```
    - Run 
        ``` docker run --rm -i -t claf/claf:latest /bin/bash ```


---


### Machine

- Machine Architecture


![images](images/claf-machine.001.png)

#### Usage

- Define the config file (.json) like [BaseConfig](#baseconfig) in `machine_config/` directory
- Run CLaF Machine (skip `/machine_config` path)


```
✗ python machine.py --machine_config {machine_config}
```


* The list of pre-defined `Machine`:

```
Machine Config:
  --machine_config MACHINE_CONFIG
    Use pre-defined machine_config (.json (.json))

    ['ko_wiki', 'nlu']
```

#### Open QA (DrQA Style)

DrQA is a system for reading comprehension applied to open-domain question answering. The system has to combine the challenges of document retrieval (finding the relevant documents) with that of machine comprehension of text (identifying the answers from those documents).

- ko_wiki: Korean Wiki Version

``` 
✗ python machine.py --machine_config ko_wiki
...
Completed!
Question > 동학의 2대 교주 이름은?
--------------------------------------------------
Doc Scores:
 - 교주 : 0.5347289443016052
 - 이교주 : 0.4967213571071625
 - 교주도 : 0.49036136269569397
 - 동학 : 0.4800325632095337
 - 동학중학교 : 0.4352934956550598
--------------------------------------------------
Answer: [
    {
        "text": "최시형",
        "score": 11.073444366455078
    },
    {
        "text": "충주목",
        "score": 9.443866729736328
    },
    {
        "text": "반월동",
        "score": 9.37778091430664
    },
    {
        "text": "환조 이자춘",
        "score": 4.64817476272583
    },
    {
        "text": "합포군",
        "score": 3.3186707496643066
    }
]
```

#### NLU (Dialog)

The reason why NLU machine does not return the full response is that response generation may require various task-specific post-processing techniques or additional logic(e.g. API calls, template-decision rules, template filling rules, nn-based response generation model) Therefore, for flexible usage, NLU machine returns only the NLU result.

``` 
✗ python machine.py --machine_config nlu
...
Utterance > "looking for a flight from Boston to Seoul or Incheon"

NLU Result: {
    "intent": "flight",
    "slots": {
        "city.depart": ["Boston"],
        "city.dest": ["Seoul", "Incheon"]
    }
}
```


## Contributing

Thanks for your interest in contributing! There are many ways to contribute to this project.  
Get started [here](./CONTRIBUTING.md).

## Maintainers

CLaF is currently maintained by 

- [Dongjun Lee](https://github.com/DongjunLee) (Author)
- [Sohee Yang](https://github.com/soheeyang)
- [Minjeong Kim](https://github.com/Mjkim88)

## Citing

If you use CLaF for your work, please cite:

```bibtex
@misc{claf,
  author = {Lee, Dongjun and Yang, Sohee and Kim, Minjeong},
  title = {CLaF: Open-Source Clova Language Framework},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/naver/claf}}
}
```

We will update this bibtex with our paper.


## Acknowledgements

`docs/` directory which includes documentation created by [Sphinx](http://www.sphinx-doc.org/).

## License

MIT license

```
Copyright (c) 2019-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
```


