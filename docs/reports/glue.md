# GLUE

- [`GLUE`](https://gluebenchmark.com/): The General Language Understanding Evaluation (GLUE) benchmark is a collection of resources for training, evaluating, and analyzing natural language understanding systems. 

---

## Results

### Dev Set

- **Base** Size : 12-layer, 768-hidden, 12-heads, 110M parameters

| Task (Metric) | Model | CLaF Result | Official Result | BaseConfig | 
| ------------- | ----- | ----- | -------- | ---------- |
| **CoLA** (**Matthew's Corr**) | BERT-Base | 59.393 | 52.1 (Test set) | glue/cola_bert.json |
|  | MT-DNN (BERT) Base | 54.658 | - | 1. multi_task/bert_base_glue.json <br/> 2. `fine-fune` |
|  | RoBERTa-Base | 64.828 | 63.6 | glue/cola_roberta.json |
| **MNLI m/mm** (**Accuracy**) | BERT-Base | 83.923/84.306 | 84.6/83.4 (Test set) | glue/mnli{m/mm}_bert.json | 
|  | MT-DNN (BERT) Base | 84.452/84.225 | - | 1. multi_task/bert_base_glue.json <br/> 2. `fine-fune` |
|  | RoBERTa-Base | 87.305/87.236 | 87.6/- | glue/mnli{m/mm}_roberta.json |
| **MRPC** (**Accuracy/F1**) | BERT-Base | 87.5/91.282 | 88.9 (Test set) | glue/mrpc_bert.json |
|  | MT-DNN (BERT) Base | 87.5/91.005 | - | 1. multi_task/bert_base_glue.json <br/> 2. `fine-fune` |
|  | RoBERTa-Base | 88.480/91.681 | 90.2 | glue/mrpc_roberta.json |
| **QNLI** (**Accuracy**) | BERT-Base | 88.521 | 90.5 (Test set) | glue/qnli_bert.json |
|  | MT-DNN (BERT) Base | - | - | 1. multi_task/bert_base_glue.json <br/> 2. `fine-fune` |
|  | RoBERTa-Base | 90.823 | 92.8 | glue/qnli_roberta.json |
| **QQP** (**Accuracy/F1**) | BERT-Base | 90.378/87.171 | 71.2 (Test set) | glue/qqp_bert.json |
|  | MT-DNN (BERT) Base | 91.261/88.219 | - | 1. multi_task/bert_base_glue.json <br/> 2. `fine-fune` |
|  | RoBERTa-Base | 91.541/88.768 | 91.9 | glue/qqp_roberta.json |
| **RTE** (**Accuracy**) | BERT-Base | 69.314 | 66.4 (Test set) | glue/rte_bert.json |
|  | MT-DNN (BERT) Base | 79.422 | - | 1. multi_task/bert_base_glue.json <br/> 2. `fine-fune` |
|  | RoBERTa-Base | 73.646 | 78.7 | glue/rte_roberta.json |
| **SST-2** (**Accuracy**) | BERT-Base | 92.546 | 93.5 (Test set) | glue/sst_bert.json |
|  | MT-DNN (BERT) Base | 93.005 | - | 1. multi_task/bert_base_glue.json <br/> 2. `fine-fune` |
|  | RoBERTa-Base | 94.495 | 94.8 | glue/sst_roberta.json |
| **STS-B** (**Pearson/Spearman**) | BERT-Base | 88.070/87.881 | 85.8 (Test set) | glue/stsb_bert.json |
|  | MT-DNN (BERT) Base | 88.444/88.807 | - | 1. multi_task/bert_base_glue.json <br/> 2. `fine-fune` |
|  | RoBERTa-Base | 89.003/89.094 | 91.2 | glue/stsb_roberta.json |
| **WNLI** (**Accuracy**) | BERT-Base | 56.338 | 65.1 (Test set) | glue/wnli_bert.json |
|  | MT-DNN (BERT) Base | 57.746 | - | 1. multi_task/bert_base_glue.json <br/> 2. `fine-fune` |
|  | RoBERTa-Base | 60.563 | - | glue/wnli_roberta.json |


- **Large** Size : 24-layer, 1024-hidden, 16-heads, 340M parameters

| Task (Metric) | Model | CLaF Result | Official Result | BaseConfig | 
| ------------- | ----- | ----- | -------- | ---------- |
| **CoLA** (**Matthew's Corr**) | BERT-Large | 61.151 | 60.6 | glue/cola_bert.json |
|  | MT-DNN (BERT) Large | - | 63.5 | 1. multi_task/bert_large_glue.json <br/>  2. `fine-fune` |
|  | RoBERTa-Large | - | 68.0 | glue/cola_roberta.json |
| **MNLI m/mm** (**Accuracy**) | BERT-Large | - | 86.6/- | glue/mnli{m/mm}_bert.json |
|  | MT-DNN (BERT) Large | - | 87.1/86.7 | 1. multi_task/bert_large_glue.json <br/>  2. `fine-fune` |
|  | RoBERTa-Large | - | 90.2/90.2 | glue/mnli{m/mm}_roberta.json |
| **MRPC** (**Accuracy/F1**) | BERT-Large | 87.255/90.845 | 88.0 | glue/mrpc_bert.json |
|  | MT-DNN (BERT) Large | - | 91.0/87.5 | 1. multi_task/bert_large_glue.json <br/>  2. `fine-fune` |
|  | RoBERTa-Large | 90.686/93.214 | 90.9 | glue/mrpc_roberta.json |
| **QNLI** (**Accuracy**) | BERT-Large | 90.440 | 92.3 | glue/qnli_bert.json |
|  | MT-DNN (BERT) Large | - | 87.1/86.7 | 1. multi_task/bert_large_glue.json <br/>  2. `fine-fune` |
|  | RoBERTa-Large | - | 94.7 | glue/qnli_roberta.json |
| **QQP** (**Accuracy/F1**) | BERT-Large | 91.640/88.745 | 91.3 | glue/qqp_bert.json |
|  | MT-DNN (BERT) Large | - | 87.1/86.7 | 1. multi_task/bert_large_glue.json <br/>  2. `fine-fune` |
|  | RoBERTa-Large | 91.848/89.031 | 92.2 | glue/qqp_roberta.json |
| **RTE** (**Accuracy**) | BERT-Large | 69.675 | 70.4 | glue/rte_bert.json |
|  | MT-DNN (BERT) Large | - | 83.4 | 1. multi_task/bert_large_glue.json <br/>  2. `fine-fune` |
|  | RoBERTa-Large | 84.838 | 86.6 | glue/rte_roberta.json |
| **SST-2** (**Accuracy**) | BERT-Large | 93.349 | 93.2 | glue/sst_bert.json |
|  | MT-DNN (BERT) Large | - | 94.3 | 1. multi_task/bert_large_glue.json <br/>  2. `fine-fune` |
|  | RoBERTa-Large | 95.642 | 96.4 | glue/sst_roberta.json |
| **STS-B** (**Pearson/Spearman**) | BERT-Large | 90.041/89735 | 90.0 | glue/stsb_bert.json |
|  | MT-DNN (BERT) Large | - | 90.7/90.6 | 1. multi_task/bert_large_glue.json <br/>  2. `fine-fune` |
|  | RoBERTa-Large | 91.980/91.764 | 92.4 | glue/stsb_roberta.json |
| **WNLI** (**Accuracy**) | BERT-Large | 59.155 | - | glue/wnli_bert.json |
|  | MT-DNN (BERT) Large | - | - | 1. multi_task/bert_large_glue.json <br/>  2. `fine-fune` |
|  | RoBERTa-Large | - | 91.3 | - |