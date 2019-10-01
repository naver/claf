
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

    def __init__(self, token_makers, tasks, pretrained_model_name=None, dropouts=None):

        super(BertForMultiTask, self).__init__(token_makers)

        self.use_pytorch_transformers = True  # for optimizer's model parameters
        self.tasks = tasks

        assert len(tasks) == len(dropouts)

        self.curr_task_category = None
        self.curr_dataset = None

        self.shared_layers = BertModel.from_pretrained(
            pretrained_model_name, cache_dir=str(CachePath.ROOT)
        )
        self.task_specific_layers = nn.ModuleList()
        for task, dropout in zip(tasks, dropouts):
            task_layer = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.shared_layers.config.hidden_size, task["num_label"])
            )
            self.task_specific_layers.append(task_layer)

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

        outputs = self.shared_layers(
            bert_inputs, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        pooled_output = outputs[1]

        task_index = features["task_index"]

        self.curr_task_category = self.tasks[task_index]["category"]
        self.curr_dataset = self._dataset.task_datasets[task_index]

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

            logits = logits.view(-1, num_label)
            if num_label == 1:
                label_value = label_value.view(-1, 1)

            loss = criterion(logits, label_value)
            output_dict["loss"] = loss.unsqueeze(0)  # NOTE: DataParallel concat Error

        return output_dict

    @overrides
    def print_examples(self, index, inputs, predictions):
        """
        Print evaluation examples

        * Args:
            index: data index
            inputs: mini-batch inputs
            predictions: prediction dictionary consisting of
                - key: 'id' (sequence id)
                - value: dictionary consisting of
                    - class_idx

        * Returns:
            print(Sequence, Sequence Tokens, Target Class, Predicted Class)
        """

        task_index = inputs["features"]["task_index"]
        task_dataset = self._dataset.task_datasets[task_index]
        task_category = self.tasks[task_index]["category"]

        data_idx = inputs["labels"]["data_idx"][index].item()
        data_id = task_dataset.get_id(data_idx)

        helper = task_dataset.helper

        sequence_a = helper["examples"][data_id]["sequence_a"]
        sequence_a_tokens = helper["examples"][data_id]["sequence_a_tokens"]
        sequence_b = helper["examples"][data_id]["sequence_b"]
        sequence_b_tokens = helper["examples"][data_id]["sequence_b_tokens"]

        print()
        print("Task(Dataset) name:", self.tasks[task_index]["name"])
        print()
        print("- Sequence a:", sequence_a)
        print("- Sequence a Tokens:", sequence_a_tokens)
        if sequence_b:
            print("- Sequence b:", sequence_b)
            print("- Sequence b Tokens:", sequence_b_tokens)

        if task_category == self.CLASSIFICATION:
            target_class_text = helper["examples"][data_id]["class_text"]

            pred_class_idx = predictions[data_id]["class_idx"]
            pred_class_text = task_dataset.get_class_text_with_idx(pred_class_idx)

            print("- Target:")
            print("    Class:", target_class_text)
            print("- Predict:")
            print("    Class:", pred_class_text)
        elif task_category == self.REGRESSION:
            target_score = helper["examples"][data_id]["score"]
            pred_score = predictions[data_id]["score"]

            print("- Target:")
            print("    Score:", target_score)
            print("- Predict:")
            print("    Score:", pred_score)

        print()

