

from overrides import overrides
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering
import torch.nn as nn

from claf.data.data_handler import CachePath
from claf.decorator import register
from claf.model.base import ModelWithoutTokenEmbedder
from claf.model.reading_comprehension.mixin import SQuADv1


@register("model:bert_for_qa")
class BertForQA(SQuADv1, ModelWithoutTokenEmbedder):
    """
    Document Reader Model. `Span Detector`

    Implementation of model presented in
    BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
    (https://arxiv.org/abs/1810.04805)

    * Args:
        token_embedder: 'QATokenEmbedder', Used to embed the 'context' and 'question'.

    * Kwargs:
        lang_code: Dataset language code [en|ko]
        pretrained_model_name: the name of a pre-trained model
        answer_maxlen: the most probable answer span of length less than or equal to {answer_maxlen}
    """

    def __init__(self, token_makers, lang_code="en", pretrained_model_name=None, answer_maxlen=30):
        super(BertForQA, self).__init__(token_makers)

        self.lang_code = lang_code
        self.bert = True  # for optimizer's model parameters
        self.answer_maxlen = answer_maxlen

        self.model = BertForQuestionAnswering.from_pretrained(
            pretrained_model_name, cache_dir=str(CachePath.ROOT)
        )
        self.criterion = nn.CrossEntropyLoss()

    @overrides
    def forward(self, features, labels=None):
        """
        * Args:
            features: feature dictionary like below.
                {"feature_name1": {
                     "token_name1": tensor,
                     "toekn_name2": tensor},
                 "feature_name2": ...}

        * Kwargs:
            label: label dictionary like below.
                {"label_name1": tensor,
                 "label_name2": tensor}
                 Do not calculate loss when there is no label. (inference/predict mode)

        * Returns: output_dict (dict) consisting of
            - start_logits: representing unnormalized log probabilities of the span start position.
            - end_logits: representing unnormalized log probabilities of the span end position.
            - best_span: the string from the original passage that the model thinks is the best answer to the question.
            - answer_idx: the question id, mapping with answer
            - loss: A scalar loss to be optimised.
        """

        bert_inputs = features["bert_input"]["feature"]
        token_type_ids = features["token_type"]["feature"]
        attention_mask = (bert_inputs > 0).long()

        span_start_logits, span_end_logits = self.model(
            bert_inputs, token_type_ids=token_type_ids, attention_mask=attention_mask
        )

        output_dict = {
            "start_logits": span_start_logits,
            "end_logits": span_end_logits,
            "best_span": self.get_best_span(
                span_start_logits, span_end_logits, answer_maxlen=self.answer_maxlen
            ),
        }

        if labels:
            answer_idx = labels["answer_idx"]
            answer_start_idx = labels["answer_start_idx"]
            answer_end_idx = labels["answer_end_idx"]

            output_dict["answer_idx"] = answer_idx

            # If we are on multi-GPU, split add a dimension
            if len(answer_start_idx.size()) > 1:
                answer_start_idx = answer_start_idx.squeeze(-1)
            if len(answer_end_idx.size()) > 1:
                answer_end_idx = answer_end_idx.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = span_start_logits.size(1)

            answer_start_idx.clamp_(0, ignored_index)
            answer_end_idx.clamp_(0, ignored_index)

            # Loss
            criterion = nn.CrossEntropyLoss(ignore_index=ignored_index)
            loss = criterion(span_start_logits, answer_start_idx)
            loss += criterion(span_end_logits, answer_end_idx)
            loss /= 2  # (start + end)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def make_metrics(self, predictions):
        """ BERT predictions need to get nbest result """

        best_predictions = {}
        for index, prediction in predictions.items():
            qid = self._dataset.get_qid(index)

            predict_text = prediction["predict_text"]

            start_logit = prediction["start_logits"][prediction["pred_span_start"]]
            end_logit = prediction["end_logits"][prediction["pred_span_end"]]
            predict_score = start_logit.item() + end_logit.item()

            if qid not in best_predictions:
                best_predictions[qid] = []
            best_predictions[qid].append((predict_text, predict_score))

        for qid, predictions in best_predictions.items():
            sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
            best_predictions[qid] = sorted_predictions[0][0]

        self.write_predictions(best_predictions)
        return self._make_metrics_with_official(best_predictions)

    @overrides
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

        context_text = arguments["context"]
        bert_tokens = helper["bert_token"]
        predictions = [
            (best_span, start_logits, end_logits)
            for best_span, start_logits, end_logits in zip(
                list(output_dict["best_span"].data),
                list(output_dict["start_logits"].data),
                list(output_dict["end_logits"].data),
            )
        ]

        best_predictions = []
        for index, prediction in enumerate(predictions):
            bert_token = bert_tokens[index]
            best_span, start_logits, end_logits = prediction
            pred_start, pred_end = best_span

            predict_text = ""
            if (
                pred_start < len(bert_token)
                and pred_end < len(bert_token)
                and bert_token[pred_start].text_span is not None
                and bert_token[pred_end].text_span is not None
            ):
                char_start = bert_token[pred_start].text_span[0]
                char_end = bert_token[pred_end].text_span[1]
                predict_text = context_text[char_start:char_end]

            start_logit = start_logits[pred_start]
            end_logit = end_logits[pred_end]
            predict_score = start_logit.item() + end_logit.item()

            best_predictions.append((predict_text, predict_score))

        sorted_predictions = sorted(best_predictions, key=lambda x: x[1], reverse=True)
        return {"text": sorted_predictions[0][0], "score": sorted_predictions[0][1]}
