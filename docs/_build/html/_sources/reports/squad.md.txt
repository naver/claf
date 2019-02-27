# SQuAD

`Span Detector`, `No Answer`

- [`SQuAD`](https://rajpurkar.github.io/SQuAD-explorer/): Stanford Question Answering Dataset (SQuAD), a new reading comprehension dataset consisting of 100,000+ questions posed by crowdworkers on a set of Wikipedia articles, where the answer to each question is a segment of text from the corresponding reading passage.
    - v1.1
    	- Train: 87599 / Dev: 10570 / Test: 9533
	- v2.0 + no_answer
	    - Train : 130319 / Dev: 11873 / Test: 8862

---

## Results (v1.1)

- Dev Set

| Model | EM (paper) | F1 (paper) | BaseConfig | Note |
| --- | --- | --- | --- | --- |
| **BiDAF** | 68.108 (67.7) | 77.780 (77.3) | squad/bidaf.json | - |
| **BiDAF + ELMo** | 74.295 | 82.727 | squad/bidaf+elmo.json | - |
| **DrQA** | 68.316 (68.8) | 77.493 (78.0) | squad/drqa.json | - |
| **DocQA** | 71.760 (71.513) | 80.635 (80.422) | squad/docqa.json | - |
| **DocQA + ELMo** | 76.244 (77.5) | 84.372 (84.5) | squad/docqa+elmo.json | - |
| **QANet** | 70.918 (73.6) | 79.800 (82.7) | squad/qanet.json | - |
| **BERT**-Base Uncased | 79.395 (80.8) | 87.130 (88.5) | squad/bert_base_uncased.json | - |
| **BERT**-Large Uncased | - (84.1) | - (90.9) | squad/bert_large_uncased.json | - |


---


## Results (v2.0)

- Dev Set

| Model | EM (paper) | F1 (paper) | BaseConfig | Note |
| --- | --- | --- | --- | --- |
| **BiDAF** | 62.570 | 65.461 | squad/bidaf_no_answer.json | - |
| **DocQA** | 61.728 | 64.489 | squad/docqa_no_answer.json | - |