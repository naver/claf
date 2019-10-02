# GLUE

- [`GLUE`](https://gluebenchmark.com/): The General Language Understanding Evaluation (GLUE) benchmark is a collection of resources for training, evaluating, and analyzing natural language understanding systems. 

---

## Results

- Dev Set

| Task (Metric) | Model | CLaF Result | Official Result | BaseConfig | 
| ------------- | ----- | ----- | -------- | ---------- |
| **CoLA** (**Matthew's Corr**) | BERT-Base | 59.393 | 52.1 (Test set) | glue/cola_bert_base.json |
|  | BERT-Large | 61.151 | 60.6 | - |
|  | MT-DNN (BERT) Base | 54.658 | - | multi_task/glue.json -> `fine-fune` |
|  | MT-DNN (BERT) Large | - | 63.5 | - |
|  | RoBERTa-Base | 64.828 | 63.6 | glue/cola_roberta_base.json |
|  | RoBERTa-Large | - | 68.0 | - |
| **MNLI m/mm** (**Accuracy**) | BERT-Base | 83.923/84.306 | 84.6/83.4 (Test set) | glue/mnli{m/mm}_bert_base.json | 
|  | BERT-Large | - | 86.6/- | - |
|  | MT-DNN (BERT) Base | 84.452/84.225 | - | multi_task/glue.json -> `fine-fune` |
|  | MT-DNN (BERT) Large | - | 87.1/86.7 | - |
|  | RoBERTa-Base | 87.305/87.236 | 87.6/- | glue/mnli{m/mm}_roberta_base.json |
|  | RoBERTa-Large | - | 90.2/90.2 | - |
| **MRPC** (**Accuracy/F1**) | BERT-Base | 87.5/91.282 | 88.9 (Test set) | glue/mrpc_bert_base.json |
|  | BERT-Large | - | 88.0 | - |
|  | MT-DNN (BERT) Base | 87.5/91.005 | - | multi_task/glue.json -> `fine-fune` |
|  | MT-DNN (BERT) Large | - | 91.0/87.5 | - |
|  | RoBERTa-Base | 88.480/91.681 | 90.2 | glue/mrpc_roberta_base.json |
|  | RoBERTa-Large | 90.686/93.214 | 90.9 | - |
| **QNLI** (**Accuracy**) | BERT-Base | 88.521 | 90.5 (Test set) | glue/qnli_bert_base.json |
|  | BERT-Large | - | 92.3 | - |
|  | MT-DNN (BERT) Base | - | - | multi_task/glue.json -> `fine-fune` |
|  | MT-DNN (BERT) Large | - | 87.1/86.7 | - |
|  | RoBERTa-Base | 90.823 | 92.8 | glue/qnli_roberta_base.json |
|  | RoBERTa-Large | - | 94.7 | - |
| **QQP** (**Accuracy/F1**) | BERT-Base | 90.378/87.171 | 71.2 (Test set) | glue/qqp_bert_base.json |
|  | BERT-Large | - | 91.3 | - |
|  | MT-DNN (BERT) Base | 91.261/88.219 | - | - |
|  | MT-DNN (BERT) Large | - | 87.1/86.7 | - |
|  | RoBERTa-Base | 91.541/88.768 | 91.9 | glue/qqp_roberta_base.json |
|  | RoBERTa-Large | - | 92.2 | - |
| **RTE** (**Accuracy**) | BERT-Base | 69.314 | 66.4 (Test set) | glue/rte_bert_base.json |
|  | BERT-Large | 69.675 | 70.4 | - |
|  | MT-DNN (BERT) Base | 79.422 | - | multi_task/glue.json -> `fine-fune` |
|  | MT-DNN (BERT) Large | - | 83.4 | - |
|  | RoBERTa-Base | 73.646 | 78.7 | glue/rte_roberta_base.json |
|  | RoBERTa-Large | 84.838 | 86.6 | - |
| **SST-2** (**Accuracy**) | BERT-Base | 92.546 | 93.5 (Test set) | glue/sst_bert_base.json |
|  | BERT-Large | 93.349 | 93.2 | - |
|  | MT-DNN (BERT) Base | 93.005 | - | - |
|  | MT-DNN (BERT) Large | - | 94.3 | - |
|  | RoBERTa-Base | 94.495 | 94.8 | glue/sst_roberta_base.json |
|  | RoBERTa-Large | - | 96.4 | - |
| **STS-B** (**Pearson/Spearman**) | BERT-Base | 88.070/87.881 | 85.8 (Test set) | glue/stsb_bert_base.json |
|  | BERT-Large | 90.041/89735 | 90.0 | - |
|  | MT-DNN (BERT) Base | 88.444/88.807 | - | multi_task/glue.json -> `fine-fune` |
|  | MT-DNN (BERT) Large | - | 90.7/90.6 | - |
|  | RoBERTa-Base | 89.003/89.094 | 91.2 | glue/stsb_roberta_base.json |
|  | RoBERTa-Large | 91.980/91.764 | 92.4 | - |
| **WNLI** (**Accuracy**) | BERT-Base | 56.338 | 65.1 (Test set) | glue/wnli_bert_base.json |
|  | BERT-Large | 59.155 | - | - |
|  | MT-DNN (BERT) Base | 57.746 | - | multi_task/glue.json -> `fine-fune` |
|  | MT-DNN (BERT) Large | - | - | - |
|  | RoBERTa-Base | 60.563 | - | glue/wnli_roberta_base.json |
|  | RoBERTa-Large | - | 91.3 | - |