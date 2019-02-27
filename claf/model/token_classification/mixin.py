
from pathlib import Path
import logging

import numpy as np
import torch
import pycm
from pycm.pycm_obj import pycmVectorError

from claf.decorator import arguments_required
import claf.utils as common_utils
from claf.model import cls_utils
from claf.metric.classification import macro_f1, macro_precision, macro_recall
from seqeval.metrics import accuracy_score as conlleval_accuracy
from seqeval.metrics import f1_score as conlleval_f1

logger = logging.getLogger(__name__)


class TokenClassification:
    """ Token Classification Mixin Class """

    def make_predictions(self, output_dict):
        """
        Make predictions with model's output_dict

        * Args:
            output_dict: model's output dictionary consisting of
                - sequence_embed: embedding vector of the sequence
                - tag_logits: representing unnormalized log probabilities of the tag

                - tag_idxs: target tag idxs
                - data_idx: data idx
                - loss: a scalar loss to be optimized

        * Returns:
            predictions: prediction dictionary consisting of
                - key: 'id' (sequence id)
                - value: dictionary consisting of
                    - tag_idxs
        """

        data_indices = output_dict["data_idx"]
        pred_tag_logits = output_dict["tag_logits"]
        pred_tag_idxs = [
            torch.argmax(pred_tag_logit, dim=-1).tolist() for pred_tag_logit in pred_tag_logits
        ]

        predictions = {
            self._dataset.get_id(data_idx.item()): {"tag_idxs": pred_tag_idx}
            for data_idx, pred_tag_idx in zip(list(data_indices.data), pred_tag_idxs)
        }

        return predictions

    @arguments_required(["sequence"])
    def predict(self, output_dict, arguments, helper):
        """
        Inference by raw_feature

        * Args:
            output_dict: model's output dictionary consisting of
                - sequence_embed: embedding vector of the sequence
                - tag_logits: representing unnormalized log probabilities of the tags.
            arguments: arguments dictionary consisting of user_input
            helper: dictionary to get the classification result, consisting of
                - tag_idx2text: dictionary converting tag_idx to tag_text

        * Returns: output dict (dict) consisting of
            - tag_logits: representing unnormalized log probabilities of the tags
            - tag_idxs: predicted tag idxs
            - tag_texts: predicted tag texts
            - tag_slots: predicted tag slots
        """

        sequence = arguments["sequence"]
        tag_logits = output_dict["tag_logits"][0]
        tag_idxs = [tag_logit.argmax(dim=-1) for tag_logit in tag_logits]
        tag_texts = [helper["tag_idx2text"][tag_idx.item()] for tag_idx in tag_idxs]

        return {
            "tag_logits": tag_logits,
            "tag_idxs": tag_idxs,
            "tag_texts": tag_texts,
            "tag_dict": cls_utils.get_tag_dict(sequence, tag_texts),
        }

    def make_metrics(self, predictions):
        """
        Make metrics with prediction dictionary

        * Args:
            predictions: prediction dictionary consisting of
                - key: 'id' (sequence id)
                - value: dictionary consisting of
                    - tag_idxs

        * Returns:
            metrics: metric dictionary consisting of
                - 'accuracy': sequence level accuracy
                - 'tag_accuracy': tag level accuracy
                - 'macro_f1': tag prediction macro(unweighted mean) f1
                - 'macro_precision': tag prediction macro(unweighted mean) precision
                - 'macro_recall': tag prediction macro(unweighted mean) recall
        """

        pred_tag_idxs_list = []
        target_tag_idxs_list = []

        accurate_sequence = []

        for data_idx, pred in predictions.items():
            target = self._dataset.get_ground_truth(data_idx)

            pred_tag_idxs_list.append(pred["tag_idxs"])
            target_tag_idxs_list.append(target["tag_idxs"])

            accurate_sequence.append(
                1 if (np.asarray(target["tag_idxs"]) == np.asarray(pred["tag_idxs"])).all() else 0
            )

        pred_tags = [
            [self._dataset.tag_idx2text[tag_idx] for tag_idx in tag_idxs] for tag_idxs in pred_tag_idxs_list
        ]
        target_tags = [
            [self._dataset.tag_idx2text[tag_idx] for tag_idx in tag_idxs] for tag_idxs in target_tag_idxs_list
        ]

        flat_pred_tags = list(common_utils.flatten(pred_tags))
        flat_target_tags = list(common_utils.flatten(target_tags))

        # confusion matrix
        try:
            pycm_obj = pycm.ConfusionMatrix(actual_vector=flat_target_tags, predict_vector=flat_pred_tags)
        except pycmVectorError as e:
            if str(e) == "Number of the classes is lower than 2":
                logger.warning("Number of tags in the batch is 1. Sanity check is highly recommended.")
                return {
                    "accuracy": 1.,
                    "tag_accuracy": 1.,

                    "macro_f1": 1.,
                    "macro_precision": 1.,
                    "macro_recall": 1.,

                    "conlleval_accuracy": 1.,
                    "conlleval_f1": 1.,
                }
            raise

        self.write_predictions(
            {"target": flat_target_tags, "predict": flat_pred_tags}, pycm_obj=pycm_obj
        )

        sequence_accuracy = sum(accurate_sequence) / len(accurate_sequence)

        metrics = {
            "accuracy": sequence_accuracy,
            "tag_accuracy": pycm_obj.Overall_ACC,

            "macro_f1": macro_f1(pycm_obj),
            "macro_precision": macro_precision(pycm_obj),
            "macro_recall": macro_recall(pycm_obj),

            "conlleval_accuracy": conlleval_accuracy(target_tags, pred_tags),
            "conlleval_f1": conlleval_f1(target_tags, pred_tags),
        }

        return metrics

    def write_predictions(self, predictions, file_path=None, is_dict=True, pycm_obj=None):
        """
        Override write_predictions() in ModelBase to log confusion matrix
        """

        super(TokenClassification, self).write_predictions(
            predictions, file_path=file_path, is_dict=is_dict
        )

        data_type = "train" if self.training else "valid"

        if pycm_obj is not None:
            stats_file_path = f"predictions-{data_type}-{self._train_counter.get_display()}-stats"
            pycm_obj.save_csv(str(Path(self._log_dir) / "predictions" / stats_file_path))

            confusion_matrix_file_path = (
                f"predictions-{data_type}-{self._train_counter.get_display()}-confusion_matrix"
            )
            cls_utils.write_confusion_matrix_to_csv(
                str(Path(self._log_dir) / "predictions" / confusion_matrix_file_path), pycm_obj
            )

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
            print(Sequence, Target Tags, Target Slots, Predicted Tags, Predicted Slots)
        """

        data_idx = inputs["labels"]["data_idx"][index].item()
        data_id = self._dataset.get_id(data_idx)

        helper = self._dataset.helper
        sequence = helper["examples"][data_id]["sequence"]
        target_tag_texts = helper["examples"][data_id]["tag_texts"]

        pred_tag_idxs = predictions[data_id]["tag_idxs"]
        pred_tag_texts = self._dataset.get_tag_texts_with_idxs(pred_tag_idxs)

        print()
        print("- Sequence:", sequence)
        print("- Target:")
        print("    Tags:", target_tag_texts)
        print("    (Slots)", cls_utils.get_tag_dict(sequence, target_tag_texts))
        print("- Predict:")
        print("    Tags:", pred_tag_texts)
        print("    (Slots)", cls_utils.get_tag_dict(sequence, pred_tag_texts))
        print()
