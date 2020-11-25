
from overrides import overrides
import torch.nn as nn
from transformers import BertModel

from claf.data.data_handler import CachePath
from claf.decorator import register
from claf.model.base import ModelWithoutTokenEmbedder
from claf.model.multi_task.category import TaskCategory
from claf.model.multi_task.mixin import MultiTask
from claf.model.reading_comprehension.mixin import ReadingComprehension


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

        self.use_transformers = True  # for optimizer's model parameters
        self.tasks = tasks

        assert len(tasks) == len(dropouts)

        self.curr_task_category = None
        self.curr_dataset = None

        self.shared_layers = BertModel.from_pretrained(
            pretrained_model_name, cache_dir=str(CachePath.ROOT)
        )
        self._init_task_layers(tasks, dropouts)
        self._init_criterions(tasks)

    def _init_criterions(self, tasks):
        self.criterions = {}
        for task_index, task in enumerate(tasks):
            task_category = task["category"]

            criterion = None
            if task_category == TaskCategory.SEQUENCE_CLASSIFICATION or task_category == TaskCategory.READING_COMPREHENSION:
                criterion = nn.CrossEntropyLoss()
            elif task_category == TaskCategory.TOKEN_CLASSIFICATION:
                ignore_tag_idx = task.get("ignore_tag_idx", 0)
                criterion = nn.CrossEntropyLoss(ignore_index=ignore_tag_idx)
            elif task_category == TaskCategory.REGRESSION:
                criterion = nn.MSELoss()
            else:
                raise ValueError("Check task_category.")

            self.criterions[task_index] = criterion

    def _init_task_layers(self, tasks, dropouts):
        self.task_specific_layers = nn.ModuleList()
        for task, dropout in zip(tasks, dropouts):
            task_category = task["category"]

            if task_category == TaskCategory.SEQUENCE_CLASSIFICATION \
                    or task_category == TaskCategory.REGRESSION:
                task_layer = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(self.shared_layers.config.hidden_size, task["num_label"])
                )
            elif task_category == TaskCategory.READING_COMPREHENSION:
                task_layer = nn.Linear(
                    self.shared_layers.config.hidden_size,
                    self.shared_layers.config.num_labels,
                )
            elif task_category == TaskCategory.TOKEN_CLASSIFICATION:
                raise NotImplementedError()
            else:
                raise ValueError("Check task_category.")

            self.task_specific_layers.append(task_layer)

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

        task_index = features["task_index"]

        self.curr_task_category = self.tasks[task_index]["category"]
        self.curr_dataset = self._dataset.task_datasets[task_index]

        bert_inputs = features["bert_input"]["feature"]
        token_type_ids = features["token_type"]["feature"]
        attention_mask = (bert_inputs > 0).long()

        shared_outputs = self.shared_layers(
            bert_inputs, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        output_dict = self._task_forward(task_index, shared_outputs)

        if labels:
            loss = self._task_calculate_loss(task_index, output_dict, labels)
            output_dict["loss"] = loss.unsqueeze(0)  # NOTE: DataParallel concat Error

        return output_dict

    def _task_forward(self, task_index, shared_outputs):
        sequence_output = shared_outputs[0]
        pooled_output = shared_outputs[1]

        task_specific_layer = self.task_specific_layers[task_index]

        task_category = self.curr_task_category
        if task_category == TaskCategory.SEQUENCE_CLASSIFICATION \
                or task_category == TaskCategory.REGRESSION:
            logits = task_specific_layer(pooled_output)

            output_dict = {
                "sequence_embed": pooled_output,
                "logits": logits,
            }
        elif task_category == TaskCategory.READING_COMPREHENSION:
            logits = task_specific_layer(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            span_start_logits = start_logits.squeeze(-1)
            span_end_logits = end_logits.squeeze(-1)

            output_dict = {
                "start_logits": span_start_logits,
                "end_logits": span_end_logits,
                "best_span": ReadingComprehension().get_best_span(
                    span_start_logits, span_end_logits, answer_maxlen=30,
                ),
            }
        elif task_category == TaskCategory.TOKEN_CLASSIFICATION:
            raise NotImplementedError()
        else:
            raise ValueError(f"Check {self.curr_task_category}.")

        output_dict["task_index"] = task_index
        return output_dict

    def _task_calculate_loss(self, task_index, output_dict, labels):
        # Loss
        num_label = self.tasks[task_index]["num_label"]
        criterion = self.criterions[task_index.item()]

        task_category = self.curr_task_category
        if task_category == TaskCategory.SEQUENCE_CLASSIFICATION \
                or task_category == TaskCategory.REGRESSION:
            label_key = None
            if task_category == TaskCategory.SEQUENCE_CLASSIFICATION:
                label_key = "class_idx"
            elif task_category == TaskCategory.REGRESSION:
                label_key = "score"

            label_value = labels[label_key]
            data_idx = labels["data_idx"]

            output_dict[label_key] = label_value
            output_dict["data_idx"] = data_idx

            logits = output_dict["logits"]
            logits = logits.view(-1, num_label)
            if num_label == 1:
                label_value = label_value.view(-1, 1)

            loss = criterion(logits, label_value)

        elif task_category == TaskCategory.READING_COMPREHENSION:
            data_idx = labels["data_idx"]
            answer_start_idx = labels["answer_start_idx"]
            answer_end_idx = labels["answer_end_idx"]

            output_dict["data_idx"] = data_idx

            # If we are on multi-GPU, split add a dimension
            if len(answer_start_idx.size()) > 1:
                answer_start_idx = answer_start_idx.squeeze(-1)
            if len(answer_end_idx.size()) > 1:
                answer_end_idx = answer_end_idx.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = output_dict["start_logits"].size(1)

            answer_start_idx.clamp_(0, ignored_index)
            answer_end_idx.clamp_(0, ignored_index)

            # Loss
            criterion = nn.CrossEntropyLoss(ignore_index=ignored_index)
            loss = criterion(output_dict["start_logits"], answer_start_idx)
            loss += criterion(output_dict["end_logits"], answer_end_idx)
            loss /= 2  # (start + end)

        elif task_category == TaskCategory.TOKEN_CLASSIFICATION:
            raise NotImplementedError()
        else:
            raise ValueError(f"Check {self.curr_task_category}.")

        return loss

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

        if task_category == TaskCategory.SEQUENCE_CLASSIFICATION \
                or task_category == TaskCategory.REGRESSION:

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

            if task_category == TaskCategory.SEQUENCE_CLASSIFICATION:
                target_class_text = helper["examples"][data_id]["class_text"]

                pred_class_idx = predictions[data_id]["class_idx"]
                pred_class_text = task_dataset.get_class_text_with_idx(pred_class_idx)

                print("- Target:")
                print("    Class:", target_class_text)
                print("- Predict:")
                print("    Class:", pred_class_text)
            elif task_category == TaskCategory.REGRESSION:
                target_score = helper["examples"][data_id]["score"]
                pred_score = predictions[data_id]["score"]

                print("- Target:")
                print("    Score:", target_score)
                print("- Predict:")
                print("    Score:", pred_score)
        elif task_category == TaskCategory.READING_COMPREHENSION:
            context = helper["examples"][data_id]["context"]
            question = helper["examples"][data_id]["question"]
            answers = helper["examples"][data_id]["answers"]

            predict_text = predictions[data_idx]["predict_text"]

            print()
            print("- Context:", context)
            print("- Question:", question)
            print("- Answers:", answers)
            print("- Predict:", predict_text)

        print()

