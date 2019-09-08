# Dataset and Model

- Reading Comprehension
- Regression
- Semantic Parsing
- Sequence Classification
- Token Classification

---

## Reading Comprehension

### Dataset

- [HistoryQA](https://oss.navercorp.com/ClovaAI-PJT/HistoryQA): Joseon History Question Answering Dataset (SQuAD Style)
- [KorQuAD](https://korquad.github.io/): KorQuAD는 한국어 Machine Reading Comprehension을 위해 만든 데이터셋입니다. 모든 질의에 대한 답변은 해당 Wikipedia 아티클 문단의 일부 하위 영역으로 이루어집니다. Stanford Question Answering Dataset(SQuAD) v1.0과 동일한 방식으로 구성되었습니다.
- [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/): **S**tanford **Qu**estion **A**nswering **D**ataset is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

### Model

- BiDAF: [Birectional Attention Flow for Machine Comprehension](https://arxiv.org/abs/1611.01603) + `No Answer`
- [A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)
- DrQA: [Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/abs/1704.00051)
- DocQA: [Simple and Effective Multi-Paragraph Reading Comprehension](https://arxiv.org/abs/1710.10723) + `No Answer`
- [QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension ](https://arxiv.org/abs/1804.09541)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

---

## Regression

- [GLUE Benchmark](https://gluebenchmark.com/): The General Language Understanding Evaluation (GLUE) benchmark is a collection of resources for training, evaluating, and analyzing natural language understanding systems.
    - STS-B

### Model

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)

---


## Semantic Parsing

### Dataset

- [WikiSQL](https://github.com/salesforce/WikiSQL): A large crowd-sourced dataset for developing natural language interfaces for relational databases. WikiSQL is the dataset released along with our work [Seq2SQL: Generating Structured Queries from Natural Language using Reinforcement Learning](http://arxiv.org/abs/1709.00103).


### Model

- SQLNet: [SQLNet: Generating Structured Queries From Natural Language Without Reinforcement Learning](https://arxiv.org/abs/1711.04436)

---


## Sequence Classification

### Dataset

- [GLUE Benchmark](https://gluebenchmark.com/): The General Language Understanding Evaluation (GLUE) benchmark is a collection of resources for training, evaluating, and analyzing natural language understanding systems.
    - CoLA, MNLI, MRPC, QNLI, QQP, RTE, SST-2, WNLI 

### Model

- [A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)

---

## Token Classification

### Dataset

- [NER - CoNLL 2013](https://www.clips.uantwerpen.be/conll2003/ner/): The shared task of CoNLL-2003 concerns language-independent named entity recognition. Named entities are phrases that contain the names of persons, organizations, locations, times and quantities. 

### Model

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)