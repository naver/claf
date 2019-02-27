
from pathlib import Path
import logging

import torch
import pycm
from pycm.pycm_obj import pycmVectorError

from claf.model import cls_utils
from claf.metric.classification import macro_f1, macro_precision, macro_recall

logger = logging.getLogger(__name__)


class SequenceClassification:
    """ Sequence Classification Mixin Class """

    def make_predictions(self, output_dict):
        """
        Make predictions with model's output_dict

        * Args:
            output_dict: model's output dictionary consisting of
                - sequence_embed: embedding vector of the sequence
                - class_logits: representing unnormalized log probabilities of the class

                - class_idx: target class idx
                - data_idx: data idx
                - loss: a scalar loss to be optimized

        * Returns:
            predictions: prediction dictionary consisting of
                - key: 'id' (sequence id)
                - value: dictionary consisting of
                    - class_idx
        """

        data_indices = output_dict["data_idx"]
        pred_class_logits = output_dict["class_logits"]
        pred_class_idxs = torch.argmax(pred_class_logits, dim=-1)

        predictions = {
            self._dataset.get_id(data_idx.item()): {"class_idx": pred_class_idx.item()}
            for data_idx, pred_class_idx in zip(list(data_indices.data), list(pred_class_idxs.data))
        }

        return predictions

    def predict(self, output_dict, arguments, helper):
        """
        Inference by raw_feature

        * Args:
            output_dict: model's output dictionary consisting of
                - sequence_embed: embedding vector of the sequence
                - class_logits: representing unnormalized log probabilities of the class.
            arguments: arguments dictionary consisting of user_input
            helper: dictionary to get the classification result, consisting of
                - class_idx2text: dictionary converting class_idx to class_text

        * Returns: output dict (dict) consisting of
            - class_logits: representing unnormalized log probabilities of the class
            - class_idx: predicted class idx
            - class_text: predicted class text
        """

        class_logits = output_dict["class_logits"]
        class_idx = class_logits.argmax(dim=-1)

        return {
            "class_logits": class_logits,
            "class_idx": class_idx,
            "class_text": helper["class_idx2text"][class_idx.item()],
        }

    def make_metrics(self, predictions):
        """
        Make metrics with prediction dictionary

        * Args:
            predictions: prediction dictionary consisting of
                - key: 'id' (sequence id)
                - value: dictionary consisting of
                    - class_idx

        * Returns:
            metrics: metric dictionary consisting of
                - 'macro_f1': class prediction macro(unweighted mean) f1
                - 'macro_precision': class prediction macro(unweighted mean) precision
                - 'macro_recall': class prediction macro(unweighted mean) recall
                - 'accuracy': class prediction accuracy
        """

        pred_classes = []
        target_classes = []

        for data_id, pred in predictions.items():
            target = self._dataset.get_ground_truth(data_id)

            pred_classes.append(self._dataset.class_idx2text[pred["class_idx"]])
            target_classes.append(target["class_text"])

        # confusion matrix
        try:
            pycm_obj = pycm.ConfusionMatrix(
                actual_vector=target_classes, predict_vector=pred_classes
            )
        except pycmVectorError as e:
            if str(e) == "Number of the classes is lower than 2":
                logger.warning("Number of classes in the batch is 1. Sanity check is highly recommended.")
                return {
                    "macro_f1": 1.,
                    "macro_precision": 1.,
                    "macro_recall": 1.,
                    "accuracy": 1.,
                }
            raise

        self.write_predictions(
            {"target": target_classes, "predict": pred_classes}, pycm_obj=pycm_obj
        )

        metrics = {
            "macro_f1": macro_f1(pycm_obj),
            "macro_precision": macro_precision(pycm_obj),
            "macro_recall": macro_recall(pycm_obj),
            "accuracy": pycm_obj.Overall_ACC,
        }

        return metrics

    def write_predictions(self, predictions, file_path=None, is_dict=True, pycm_obj=None):
        """
        Override write_predictions() in ModelBase to log confusion matrix
        """

        super(SequenceClassification, self).write_predictions(
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
            print(Sequence, Target Class, Predicted Class)
        """

        data_idx = inputs["labels"]["data_idx"][index].item()
        data_id = self._dataset.get_id(data_idx)

        helper = self._dataset.helper
        sequence = helper["examples"][data_id]["sequence"]
        target_class_text = helper["examples"][data_id]["class_text"]

        pred_class_idx = predictions[data_id]["class_idx"]
        pred_class_text = self._dataset.get_class_text_with_idx(pred_class_idx)

        print()
        print("- Sequence:", sequence)
        print("- Target:")
        print("    Class:", target_class_text)
        print("- Predict:")
        print("    Class:", pred_class_text)
        print()
