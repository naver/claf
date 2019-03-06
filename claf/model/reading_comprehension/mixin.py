
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from claf.decorator import arguments_required
from claf.metric import korquad_v1_official, squad_v1_official, squad_v2_official


class ReadingComprehension:
    """
    Reading Comprehension Mixin Class

    * Args:
        token_embedder: 'RCTokenEmbedder', Used to embed the 'context' and 'question'.

    """

    def get_best_span(self, span_start_logits, span_end_logits, answer_maxlen=None):
        """
        Take argmax of constrained score_s * score_e.

        * Args:
            span_start_logits: independent start logits
            span_end_logits: independent end logits

        * Kwargs:
            answer_maxlen: max span length to consider (default is None -> All)
        """

        B = span_start_logits.size(0)
        best_word_span = span_start_logits.new_zeros((B, 2), dtype=torch.long)

        score_starts = F.softmax(span_start_logits, dim=-1)
        score_ends = F.softmax(span_end_logits, dim=-1)

        max_len = answer_maxlen or score_starts.size(1)

        for i in range(score_starts.size(0)):
            # Outer product of scores to get full p_s * p_e matrix
            scores = torch.ger(score_starts[i], score_ends[i])

            # Zero out negative length and over-length span scores
            scores.triu_().tril_(max_len - 1)

            # Take argmax or top n
            scores = scores.detach().cpu().numpy()
            scores_flat = scores.flatten()

            idx_sort = [np.argmax(scores_flat)]

            s_idx, e_idx = np.unravel_index(idx_sort, scores.shape)
            best_word_span[i, 0] = int(s_idx[0])
            best_word_span[i, 1] = int(e_idx[0])

        return best_word_span

    def _make_span_metrics(self, predictions):
        """ span accuracy metrics """
        start_accuracy, end_accuracy, span_accuracy = 0, 0, 0

        for index, preds in predictions.items():
            _, _, (answer_start, answer_end) = self._dataset.get_ground_truths(index)

            start_acc = 1 if preds["pred_span_start"] == answer_start else 0
            end_acc = 1 if preds["pred_span_end"] == answer_end else 0
            span_acc = 1 if start_acc == 1 and end_acc == 1 else 0

            start_accuracy += start_acc
            end_accuracy += end_acc
            span_accuracy += span_acc

        start_accuracy = 100.0 * start_accuracy / len(self._dataset)
        end_accuracy = 100.0 * end_accuracy / len(self._dataset)
        span_accuracy = 100.0 * span_accuracy / len(self._dataset)

        return {"start_acc": start_accuracy, "end_acc": end_accuracy, "span_acc": span_accuracy}

    def make_predictions(self, output_dict):
        """
        Make predictions with model's output_dict

        * Args:
            output_dict: model's output dictionary consisting of
                - answer_idx: question id
                - best_span: calculate the span_start_logits and span_end_logits to what is the best span
                - start_logits: span start logits
                - end_logits: span end logits

        * Returns:
            predictions: prediction dictionary consisting of
                - key: 'id' (question id)
                - value: consisting of dictionary
                    predict_text, pred_span_start, pred_span_end, span_start_prob, span_end_prob
        """

        data_indices = output_dict["answer_idx"]
        best_word_span = output_dict["best_span"]

        return OrderedDict(
            [
                (
                    index.item(),
                    {
                        "predict_text": self._dataset.get_text_with_index(
                            index.item(), best_span[0], best_span[1]
                        ),
                        "pred_span_start": best_span[0],
                        "pred_span_end": best_span[1],
                        "start_logits": start_logits,
                        "end_logits": end_logits,
                    },
                )
                for index, best_span, start_logits, end_logits in zip(
                    list(data_indices.data),
                    list(best_word_span.data),
                    list(output_dict["start_logits"].data),
                    list(output_dict["end_logits"].data),
                )
            ]
        )

    @arguments_required(["context", "question"])
    def predict(self, output_dict, arguments, helper):
        """
        Inference by raw_feature

        * Args:
            output_dict: model's output dictionary consisting of
                - answer_idx: question id
                - best_span: calculate the span_start_logits and span_end_logits to what is the best span
            arguments: arguments dictionary consisting of user_input
            helper: dictionary for helping get answer

        * Returns:
            span: predict best_span
        """
        span_start, span_end = list(output_dict["best_span"][0].data)
        word_start = span_start.item()
        word_end = span_end.item()

        text_span = helper["text_span"]
        char_start = text_span[word_start][0]
        char_end = text_span[word_end][1]

        context_text = arguments["context"]
        answer_text = context_text[char_start:char_end]

        start_logit = output_dict["start_logits"][0]
        end_logit = output_dict["end_logits"][0]

        score = start_logit[span_start] + end_logit[span_end]
        score = score.item()

        return {"text": answer_text, "score": score}

    def print_examples(self, index, inputs, predictions):
        """
        Print evaluation examples

        * Args:
            index: data index
            inputs: mini-batch inputs
            predictions: prediction dictionary consisting of
                - key: 'id' (question id)
                - value: consisting of dictionary
                    predict_text, pred_span_start, pred_span_end, span_start_prob, span_end_prob

        * Returns:
            print(Context, Question, Answers and Predict)
        """
        data_index = inputs["labels"]["answer_idx"][index].item()
        qid = self._dataset.get_qid(data_index)
        if "#" in qid:  # bert case (qid#index)
            qid = qid.split("#")[0]

        helper = self._dataset.helper

        context = helper["examples"][qid]["context"]
        question = helper["examples"][qid]["question"]
        answers = helper["examples"][qid]["answers"]

        predict_text = predictions[data_index]["predict_text"]

        print()
        print("- Context:", context)
        print("- Question:", question)
        print("- Answers:", answers)
        print("- Predict:", predict_text)
        print()


class SQuADv1(ReadingComprehension):
    """
    Reading Comprehension Mixin Class
        with SQuAD v1.1 evaluation

    * Args:
        token_embedder: 'QATokenEmbedder', Used to embed the 'context' and 'question'.

    """

    def make_metrics(self, predictions):
        """
        Make metrics with prediction dictionary

        * Args:
            predictions: prediction dictionary consisting of
                - key: 'id' (question id)
                - value: (predict_text, pred_span_start, pred_span_end)

        * Returns:
            metrics: metric dictionary consisting of
                - 'em': exact_match (SQuAD v1.1 official evaluation)
                - 'f1': f1 (SQuAD v1.1 official evaluation)
                - 'start_acc': span_start accuracy
                - 'end_acc': span_end accuracy
                - 'span_acc': span accuracy (start and end)
        """

        preds = {}
        for index, prediction in predictions.items():
            _, _, (answer_start, answer_end) = self._dataset.get_ground_truths(index)

            qid = self._dataset.get_qid(index)
            preds[qid] = prediction["predict_text"]

        self.write_predictions(preds)

        squad_offical_metrics = self._make_metrics_with_official(preds)

        metrics = self._make_span_metrics(predictions)
        metrics.update(squad_offical_metrics)
        return metrics

    def _make_metrics_with_official(self, preds):
        """ SQuAD v1.1 official evaluation """
        dataset = self._dataset.raw_dataset

        if self.lang_code.startswith("ko"):
            scores = korquad_v1_official.evaluate(dataset, preds)
        else:
            scores = squad_v1_official.evaluate(dataset, preds)
        return scores


class SQuADv2(ReadingComprehension):
    """
    Reading Comprehension Mixin Class
        with SQuAD v2.0 evaluation

    * Args:
        token_embedder: 'RCTokenEmbedder', Used to embed the 'context' and 'question'.

    """

    def make_metrics(self, predictions):
        """
        Make metrics with prediction dictionary

        * Args:
            predictions: prediction dictionary consisting of
                - key: 'id' (question id)
                - value: consisting of dictionary
                    predict_text, pred_span_start, pred_span_end, span_start_prob, span_end_prob

        * Returns:
            metrics: metric dictionary consisting of
                - 'start_acc': span_start accuracy
                - 'end_acc': span_end accuracy
                - 'span_acc': span accuracy (start and end)
                - 'em': exact_match (SQuAD v2.0 official evaluation)
                - 'f1': f1 (SQuAD v2.0 official evaluation)
                - 'HasAns_exact': has answer exact_match
                - 'HasAns_f1': has answer f1
                - 'NoAns_exact': no answer exact_match
                - 'NoAns_f1': no answer f1
                - 'best_exact': best exact_match score with best_exact_thresh
                - 'best_exact_thresh': best exact_match answerable threshold
                - 'best_f1': best f1 score with best_f1_thresh
                - 'best_f1_thresh': best f1 answerable threshold
        """

        preds, na_probs = {}, {}
        for index, prediction in predictions.items():
            _, _, (answer_start, answer_end) = self._dataset.get_ground_truths(index)

            # Metrics (SQuAD official metric)
            predict_text = prediction["predict_text"]
            if predict_text == "<noanswer>":
                predict_text = ""

            qid = self._dataset.get_qid(index)
            preds[qid] = predict_text

            span_start_probs = F.softmax(prediction["start_logits"], dim=-1)
            span_end_probs = F.softmax(prediction["end_logits"], dim=-1)

            start_no_prob = span_start_probs[-1].item()
            end_no_prob = span_end_probs[-1].item()
            no_answer_prob = start_no_prob * end_no_prob
            na_probs[qid] = no_answer_prob

        self.write_predictions(preds)

        model_type = "train" if self.training else "valid"
        self.write_predictions(
            na_probs, file_path=f"na_probs-{model_type}-{self._train_counter.get_display()}.json"
        )

        squad_offical_metrics = self._make_metrics_with_official(preds, na_probs)

        metrics = self._make_span_metrics(predictions)
        metrics.update(squad_offical_metrics)
        return metrics

    def _make_metrics_with_official(self, preds, na_probs, na_prob_thresh=1.0):
        """ SQuAD 2.0 official evaluation """
        dataset = self._dataset.raw_dataset

        squad_scores = squad_v2_official.evaluate(dataset, na_probs, preds)
        squad_scores["em"] = squad_scores["exact"]

        remove_keys = ["total", "exact", "HasAns_total", "NoAns_total"]
        for key in remove_keys:
            if key in squad_scores:
                del squad_scores[key]

        return squad_scores
