
import logging

from claf.metric.glue import pearson_and_spearman
from claf.metric.regression import mse

logger = logging.getLogger(__name__)


class Regression:
    """ Regression Mixin Class """

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
                    - score
        """

        data_indices = output_dict["data_idx"]
        pred_logits = output_dict["logits"]

        predictions = {
            self._dataset.get_id(data_idx.item()): {"score": pred_score.item()}
            for data_idx, pred_score in zip(list(data_indices.data), list(pred_logits.data))
        }

        return predictions

    def predict(self, output_dict, arguments, helper):
        """
        Inference by raw_feature

        * Args:
            output_dict: model's output dictionary consisting of
                - sequence_embed: embedding vector of the sequence
                - logits: model's score
            arguments: arguments dictionary consisting of user_input
            helper: dictionary to get the classification result, consisting of
                 -

        * Returns: output dict (dict) consisting of
            - score: model's score
        """

        score = output_dict["logits"]

        return {
            "score": score,
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
                - 'mse': Mean Squard Error
                - 'pearson': Pearson correlation coefficient
                - 'spearmanr': Spearman correlation coefficient
                - 'pearson_spearman_corr': (pearson_corr + spearman_corr) / 2,
        """

        pred_scores = []
        target_scores = []

        preds = {}
        for data_id, pred in predictions.items():
            target = self._dataset.get_ground_truth(data_id)

            preds[data_id] = pred["score"]

            pred_scores.append(pred["score"])
            target_scores.append(target["score"])

        self.write_predictions(preds)

        metrics = {"mse": mse(pred_scores, target_scores) / len(target_scores)}

        pearson_spearman_metrics = pearson_and_spearman(pred_scores, target_scores)
        metrics.update(pearson_spearman_metrics)

        return metrics

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

        target_score = helper["examples"][data_id]["score"]
        pred_score = predictions[data_id]["score"]

        print()
        print("- Sequence:", sequence)
        print("- Target:")
        print("    Score:", target_score)
        print("- Predict:")
        print("    Score:", pred_score)
        print()
