
from overrides import overrides
from pytorch_transformers import RobertaModel
import torch.nn as nn

from claf.data.data_handler import CachePath
from claf.decorator import register
from claf.model.base import ModelWithoutTokenEmbedder
from claf.model.reading_comprehension.mixin import SQuADv1ForBert


@register("model:roberta_for_qa")
class RoBertaForQA(SQuADv1ForBert, ModelWithoutTokenEmbedder):
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
        super(RoBertaForQA, self).__init__(token_makers)

        self.lang_code = lang_code
        self.use_pytorch_transformers = True  # for optimizer's model parameters
        self.answer_maxlen = answer_maxlen

        self.model = RobertaModel.from_pretrained(
            pretrained_model_name, cache_dir=str(CachePath.ROOT)
        )
        self.qa_outputs = nn.Linear(self.model.config.hidden_size, self.model.config.num_labels)
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
            - data_idx: the question id, mapping with answer
            - loss: A scalar loss to be optimised.
        """

        bert_inputs = features["bert_input"]["feature"]
        attention_mask = (bert_inputs > 0).long()

        outputs = self.model(
            bert_inputs, token_type_ids=None, attention_mask=attention_mask
        )
        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        span_start_logits, span_end_logits = logits.split(1, dim=-1)

        span_start_logits = span_start_logits.squeeze(-1)
        span_end_logits = span_end_logits.squeeze(-1)

        output_dict = {
            "start_logits": span_start_logits,
            "end_logits": span_end_logits,
            "best_span": self.get_best_span(
                span_start_logits, span_end_logits, answer_maxlen=self.answer_maxlen
            ),
        }

        if labels:
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
