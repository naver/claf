
from overrides import overrides
from pytorch_transformers import BertModel
import torch.nn as nn

from claf.data.data_handler import CachePath
from claf.decorator import register
from claf.model.base import ModelWithoutTokenEmbedder
from claf.model.multi_task.mixin import MultiTask


@register("model:bert_for_multi")
class BertForMultiTask(MultiTask, ModelWithoutTokenEmbedder):
    """
    Implementation of Sentence Classification model presented in
    BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
    (https://arxiv.org/abs/1810.04805)

    * Args:
        token_embedder: used to embed the sequence
        num_classes: number of classified classes

    * Kwargs:
        pretrained_model_name: the name of a pre-trained model
        dropout: classification layer dropout
    """

    def __init__(self, token_makers, tasks, pretrained_model_name=None, dropout=0.2):

        super(BertForMultiTask, self).__init__(token_makers)

        self.use_pytorch_transformers = True  # for optimizer's model parameters
        self.tasks = tasks

        self.curr_task_category = None
        self.curr_dataset = None

        self._model = BertModel.from_pretrained(
            pretrained_model_name, cache_dir=str(CachePath.ROOT)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self._model.config.hidden_size, self._model.config.hidden_size),
            nn.Dropout(dropout),
        )
        self.classifier.apply(self._model.init_weights)

        self.task_specific_layers = nn.ModuleList(
            [nn.Linear(self._model.config.hidden_size, t["num_label"]) for t in tasks]
        )

        self.criterions = {
            "classification": nn.CrossEntropyLoss(),
            "regression": nn.MSELoss(),
        }

    @overrides
    def forward(self, features, labels=None):
        """
        * Args:
            features: feature dictionary like below.
            {
                "bert_input": {
                    "feature": [
                        [3, 4, 1, 0, 0, 0, ...],
                        ...,
                    ]
                },
                "token_type": {
                    "feature": [
                        [0, 0, 0, 0, 0, 0, ...],
                        ...,
                    ],
                }
            }

        * Kwargs:
            label: label dictionary like below.
            {
                "class_idx": [2, 1, 0, 4, 5, ...]
                "data_idx": [2, 4, 5, 7, 2, 1, ...]
            }
            Do not calculate loss when there is no label. (inference/predict mode)

        * Returns: output_dict (dict) consisting of
            - sequence_embed: embedding vector of the sequence
            - logits: representing unnormalized log probabilities

            - class_idx: target class idx
            - data_idx: data idx
            - loss: a scalar loss to be optimized
        """

        bert_inputs = features["bert_input"]["feature"]
        token_type_ids = features["token_type"]["feature"]
        attention_mask = (bert_inputs > 0).long()

        outputs = self._model(
            bert_inputs, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        pooled_output = outputs[1]

        task_index = features["task_index"]

        self.curr_task_category = self.tasks[task_index]["category"]
        self.curr_dataset = self._dataset.task_datasets[task_index]

        pooled_output = self.classifier(pooled_output)
        task_specific_layer = self.task_specific_layers[task_index]
        logits = task_specific_layer(pooled_output)

        output_dict = {
            "task_index": task_index,
            "sequence_embed": pooled_output,
            "logits": logits,
        }

        if labels:
            label_key = None
            if self.curr_task_category == self.CLASSIFICATION:
                label_key = "class_idx"
            elif self.curr_task_category == self.REGRESSION:
                label_key = "score"
            else:
                raise ValueError("task category error.")

            label_value = labels[label_key]
            data_idx = labels["data_idx"]

            output_dict[label_key] = label_value
            output_dict["data_idx"] = data_idx

            # Loss
            num_label = self.tasks[task_index]["num_label"]

            criterion = self.criterions[self.curr_task_category]
            loss = criterion(
                logits.view(-1, num_label), label_value.view(-1)
            )
            output_dict["loss"] = loss.unsqueeze(0)  # NOTE: DataParallel concat Error

        return output_dict

    @overrides
    def print_examples(self, index, inputs, predictions):
        print("print_examples in BertForMultiTask!")
