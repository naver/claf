# Reading Comprehension


Focus on Service orientied metrics (eg. 1-example inference latency)

- Exists samples in `reports/summary/` directory

## SQuAD v1.1


| Model | Inference Latency <br/>(1-example/ms) | F1 (SQuAD) | BaseConfig | Note |
| --- | --- | --- | --- | --- |
| **BiDAF** | 142.644 `cpu` / 32.545 `gpu` | 77.747 | squad/bidaf.json | - |
| **BiDAF + ELMo** | - `cpu` / - `gpu` | 82.288 | squad/bidaf+elmo.json | - |
| **DrQA** | - `cpu` / - `gpu` | 77.049 | squad/drqa.json | - |
| **DocQA** | - `cpu` / - `gpu` | 80.635 | squad/docqa.json | - |
| **DocQA + ELMo** | - `cpu` / - `gpu` | 84.372 | squad/docqa+elmo.json | - |
| **QANet** | - `cpu` / - `gpu` | 79.800 | squad/qanet.json | - |
| **BERT** | - `cpu` / - `gpu` | 87.130 | squad/bert\_base-_uncased.json | - |


### Plot

- Inference Latency (1-example)

![images](../../images/inference_latency_chart-1000.png)