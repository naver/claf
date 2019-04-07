
from pathlib import Path
import logging
from collections import defaultdict

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


class SequenceTokenClassification:
    """ Sequence and Token Classification Mixin Class """
    def make_predictions(self, output_dict):
        """
        Make predictions with model's output_dict

        * Args:
            output_dict: model's output dictionary consisting of
                - sequence_embed: embedding vector of the sequence
                - class_logits: representing unnormalized log probabilities of the class
                - tag_logits: representing unnormalized log probabilities of the tag

                - (pred_tag_idxs: tag idxs by CRF)

                - class_idx: target class idx
                - tag_idxs: target tag idxs
                - data_idx: data idx
                - loss: a scalar loss to be optimized

        * Returns:
            predictions: prediction dictionary consisting of
                - key: 'id' (sequence id)
                - value: dictionary consisting of
                    - class_idx
                    - tag_idxs
        """

        data_indices = output_dict["data_idx"]

        pred_class_logits = output_dict["class_logits"]
        pred_class_idxs = torch.argmax(pred_class_logits, dim=-1)

        pred_tag_logits = output_dict["tag_logits"]

        if "pred_tag_idxs" in output_dict:  # crf
            pred_tag_idxs = output_dict["pred_tag_idxs"]

        else:
            pred_tag_idxs = [
                torch.argmax(pred_tag_logit, dim=-1).data.cpu().numpy() for pred_tag_logit in pred_tag_logits
            ]

        predictions = {
            self._dataset.get_id(data_idx.item()): {"class_idx": pred_class_idx.item(), "tag_idxs": pred_tag_idx}
            for data_idx, pred_class_idx, pred_tag_idx in zip(list(data_indices.data), pred_class_idxs, pred_tag_idxs)
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
                - (pred_tag_idxs: tag idxs by CRF)
                - tag_logits: representing unnormalized log probabilities of the tags.
            arguments: arguments dictionary consisting of user_input
            helper: dictionary to get the classification result, consisting of
                - class_idx2text: dictionary converting class_idx to class_text
                - tag_idx2text: dictionary converting tag_idx to tag_text
                - remove_postpos: postprocess

        * Returns: output dict (dict) consisting of
            - class_logits: representing unnormalized log probabilities of the class
            - class_idx: predicted class idx
            - class_text: predicted class text

            - tag_logits: representing unnormalized log probabilities of the tags
            - tag_idxs: predicted tag idxs
            - tag_texts: predicted tag texts
            - tag_slots: predicted tag slots
        """

        sequence = arguments["sequence"]
        return_logits = arguments["return_logits"]

        class_logits = output_dict["class_logits"]
        class_idx = class_logits.argmax(dim=-1)

        tag_logits = output_dict["tag_logits"][0]

        if "pred_tag_idxs" in output_dict:
            tag_idxs = output_dict["pred_tag_idxs"][0]
            tag_texts = [helper["tag_idx2text"][tag_idx] for tag_idx in tag_idxs]
        else:
            tag_idxs = [tag_logit.argmax(dim=-1) for tag_logit in tag_logits]
            tag_texts = [helper["tag_idx2text"][tag_idx.item()] for tag_idx in tag_idxs]

        result_dict = {
            "sequence": sequence,

            "class_idx": class_idx,
            "class_text": helper["class_idx2text"][class_idx.item()],

            "tag_idxs": tag_idxs,
            "tag_texts": tag_texts,
            "tag_dict": cls_utils.get_tag_entities(sequence, tag_texts),
        }

        if return_logits:
            result_dict.update({
                "class_logits": class_logits,
                "tag_logits": tag_logits,
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
                    - tag_idxs

        * Returns:
            metrics: metric dictionary consisting of
                - 'class_macro_f1': class prediction macro(unweighted mean) f1
                - 'class_macro_precision': class prediction macro(unweighted mean) precision
                - 'class_macro_recall': class prediction macro(unweighted mean) recall
                - 'class_accuracy': class prediction accuracy

                - 'tag_sequence_accuracy': sequence level tag accuracy
                - 'tag_accuracy': tag level accuracy
                - 'tag_macro_f1': tag prediction macro(unweighted mean) f1
                - 'tag_macro_precision': tag prediction macro(unweighted mean) precision
                - 'tag_macro_recall': tag prediction macro(unweighted mean) recall

                - 'tag_conlleval_accuracy': tag prediction conlleval accuracy
                - 'tag_conlleval_f1': tag prediction conlleval f1
        """

        pred_classes = []
        target_classes = []

        pred_tag_idxs_list = []
        target_tag_idxs_list = []

        accurate_sequence = []

        # topk
        pred_topk = defaultdict(list)

        for data_idx, pred in predictions.items():
            class_target, tag_target = self._dataset.get_ground_truth(data_idx)

            pred_classes.append(self._dataset.class_idx2text[pred["class_idx"]])
            target_classes.append(class_target["class_text"])

            pred_tag_idxs_list.append(pred["tag_idxs"])
            target_tag_idxs_list.append(tag_target["tag_idxs"])

            accurate_sequence.append(
                1 if (np.asarray(tag_target["tag_idxs"]) == np.asarray(pred["tag_idxs"])).all() else 0
            )

            # topk
            for k in range(self.K):
                pred_topk[k + 1].append(self._dataset.class_idx2text[pred[f"top{k + 1}"]])

        pred_tags = [
            [self._dataset.tag_idx2text[tag_idx] for tag_idx in tag_idxs] for tag_idxs in pred_tag_idxs_list
        ]
        target_tags = [
            [self._dataset.tag_idx2text[tag_idx] for tag_idx in tag_idxs] for tag_idxs in target_tag_idxs_list
        ]

        flat_pred_tags = list(common_utils.flatten(pred_tags))
        flat_target_tags = list(common_utils.flatten(target_tags))

        # class confusion matrix
        try:
            class_pycm_obj = pycm.ConfusionMatrix(actual_vector=target_classes, predict_vector=pred_classes)
            self.write_predictions(
                {"target": target_classes, "predict": pred_classes}, pycm_obj=class_pycm_obj, label_type="class"
            )

            # topk
            for k in range(self.K):
                self.write_predictions(
                    {"target": target_classes, "predict": pred_topk[k + 1]}, label_type=f"class_top{k + 1}"
                )

            class_metrics = {
                "class_macro_f1": macro_f1(class_pycm_obj),
                "class_macro_precision": macro_precision(class_pycm_obj),
                "class_macro_recall": macro_recall(class_pycm_obj),
                "class_accuracy": class_pycm_obj.Overall_ACC,
            }

            # topk
            num_correct = 0
            for k in range(self.K):
                num_correct += sum(np.asarray(target_classes) == np.asarray(pred_topk[k + 1]))
                class_metrics.update({
                    f"class_accuracy_top{k + 1}": num_correct / len(target_classes),
                })

        except pycmVectorError as e:
            if str(e) == "Number of the classes is lower than 2":
                logger.warning("Number of tags in the batch is 1. Sanity check is highly recommended.")
                class_metrics = {
                    "class_macro_f1": 1.,
                    "class_macro_precision": 1.,
                    "class_macro_recall": 1.,
                    "class_accuracy": 1.,
                }
                for k in range(self.K):
                    class_metrics.update({
                        f"class_accuracy_top{k + 1}": 1.,
                    })
            else:
                raise

        # tag confusion matrix
        try:
            tag_pycm_obj = pycm.ConfusionMatrix(actual_vector=flat_target_tags, predict_vector=flat_pred_tags)
            self.write_predictions(
                {"target": flat_target_tags, "predict": flat_pred_tags}, pycm_obj=tag_pycm_obj, label_type="tag"
            )

            tag_sequence_accuracy = sum(accurate_sequence) / len(accurate_sequence)
            tag_metrics = {
                "tag_sequence_accuracy": tag_sequence_accuracy,
                "tag_accuracy": tag_pycm_obj.Overall_ACC,

                "tag_macro_f1": macro_f1(tag_pycm_obj),
                "tag_macro_precision": macro_precision(tag_pycm_obj),
                "tag_macro_recall": macro_recall(tag_pycm_obj),

                "tag_conlleval_accuracy": conlleval_accuracy(target_tags, pred_tags),
                "tag_conlleval_f1": conlleval_f1(target_tags, pred_tags),
            }

        except pycmVectorError as e:
            if str(e) == "Number of the classes is lower than 2":
                logger.warning("Number of tags in the batch is 1. Sanity check is highly recommended.")
                tag_metrics = {
                    "tag_sequence_accuracy": 1.,
                    "tag_accuracy": 1.,

                    "tag_macro_f1": 1.,
                    "tag_macro_precision": 1.,
                    "tag_macro_recall": 1.,

                    "tag_conlleval_accuracy": 1.,
                    "tag_conlleval_f1": 1.,
                }
            else:
                raise

        metrics = {}
        metrics.update(class_metrics)
        metrics.update(tag_metrics)

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

        super(SequenceTokenClassification, self).write_predictions(
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
                    - tag_idxs

        * Returns:
            print(Sequence, Target Class, Target Tags, Target Entities, Predicted Class, Predicted Tags, Predicted Entities)
        """

        data_idx = inputs["labels"]["data_idx"][index].item()
        data_id = self._dataset.get_id(data_idx)

        helper = self._dataset.helper
        sequence = helper["examples"][data_id]["sequence"]

        target_class_text = helper["examples"][data_id]["class_text"]
        target_tag_texts = helper["examples"][data_id]["tag_texts"]

        pred_class_idx = predictions[data_id]["class_idx"]
        pred_class_text = self._dataset.get_class_text_with_idx(pred_class_idx)

        pred_tag_idxs = predictions[data_id]["tag_idxs"]
        pred_tag_texts = self._dataset.get_tag_texts_with_idxs(pred_tag_idxs)

        print()
        print("- Sequence:", sequence)
        print("- Target:")
        print("    Class:", target_class_text)
        print("    Tags:", target_tag_texts)
        print("    (Entities)", cls_utils.get_tag_entities(sequence, target_tag_texts))
        print("- Predict:")
        print("    Class:", pred_class_text)
        print("    Tags:", pred_tag_texts)
        print("    (Entities)", cls_utils.get_tag_entities(sequence, pred_tag_texts))
        print()
