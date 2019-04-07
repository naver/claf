
from pathlib import Path
from collections import defaultdict
import logging

import numpy as np
import torch
import pycm
from pycm.pycm_obj import pycmVectorError

from claf.decorator import arguments_required
from claf.model import cls_utils
from claf.metric.classification import macro_f1, macro_precision, macro_recall

logger = logging.getLogger(__name__)


class SequenceClassification:
    """ Sequence Classification Mixin Class """

    K = 5  # for topk

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

        # topk
        _, topk = torch.topk(pred_class_logits, dim=-1, k=self.K)
        for k in range(self.K):
            for data_idx, pred_class_topk in zip(list(data_indices.data), topk[:, k]):
                predictions[self._dataset.get_id(data_idx.item())].update({
                    f"top{k + 1}": pred_class_topk.item()
                })

        return predictions

    @arguments_required(["sequence", "return_logits"])
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

        sequence = arguments["sequence"]
        return_logits = arguments["return_logits"]

        class_logits = output_dict["class_logits"]
        class_idx = class_logits.argmax(dim=-1)

        result_dict = {
            "sequence": sequence,

            "class_idx": class_idx,
            "class_text": helper["class_idx2text"][class_idx.item()],
        }

        if return_logits:
            result_dict.update({
                "class_logits": class_logits,
            })

        return result_dict

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
                - 'class_macro_f1': class prediction macro(unweighted mean) f1
                - 'class_macro_precision': class prediction macro(unweighted mean) precision
                - 'class_macro_recall': class prediction macro(unweighted mean) recall
                - 'class_accuracy': class prediction accuracy
        """

        pred_classes = []
        target_classes = []

        # topk
        pred_topk = defaultdict(list)

        for data_id, pred in predictions.items():
            target = self._dataset.get_ground_truth(data_id)

            pred_classes.append(self._dataset.class_idx2text[pred["class_idx"]])
            target_classes.append(target["class_text"])

            # topk
            for k in range(self.K):
                pred_topk[k + 1].append(self._dataset.class_idx2text[pred[f"top{k + 1}"]])

        # confusion matrix
        try:
            pycm_obj = pycm.ConfusionMatrix(
                actual_vector=target_classes, predict_vector=pred_classes
            )
        except pycmVectorError as e:
            if str(e) == "Number of the classes is lower than 2":
                logger.warning("Number of classes in the batch is 1. Sanity check is highly recommended.")
                return {
                    "class_macro_f1": 1.,
                    "class_macro_precision": 1.,
                    "class_macro_recall": 1.,
                    "class_accuracy": 1.,
                }
            raise

        self.write_predictions(
            {"target": target_classes, "predict": pred_classes}, pycm_obj=pycm_obj, label_type="class"
        )

        # topk
        for k in range(self.K):
            self.write_predictions(
                {"target": target_classes, "predict": pred_topk[k + 1]}, label_type=f"class_top{k + 1}"
            )

        metrics = {
            "class_macro_f1": macro_f1(pycm_obj),
            "class_macro_precision": macro_precision(pycm_obj),
            "class_macro_recall": macro_recall(pycm_obj),
            "class_accuracy": pycm_obj.Overall_ACC,
        }

        # topk
        num_correct = 0
        for k in range(self.K):
            num_correct += sum(np.asarray(target_classes) == np.asarray(pred_topk[k + 1]))
            metrics.update({
                f"class_accuracy_top{k + 1}": num_correct / len(target_classes),
            })

        return metrics

    def write_predictions(self, predictions, file_path=None, is_dict=True, pycm_obj=None, label_type=None):
        """
        Override write_predictions() in ModelBase to log confusion matrix
        """

        super(SequenceClassification, self).write_predictions(
            predictions, file_path=file_path, is_dict=is_dict
        )

        data_type = f"train-{label_type}" if self.training else f"valid-{label_type}"

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
