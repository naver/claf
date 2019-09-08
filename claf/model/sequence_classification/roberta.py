
from overrides import overrides
from pytorch_transformers import RobertaModel
import torch.nn as nn

from claf.data.data_handler import CachePath
from claf.decorator import register
from claf.model.base import ModelWithoutTokenEmbedder
from claf.model.sequence_classification.mixin import SequenceClassification


@register("model:roberta_for_seq_cls")
class RobertaForSeqCls(SequenceClassification, ModelWithoutTokenEmbedder):
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

    def __init__(self, token_makers, num_classes, pretrained_model_name=None, dropout=0.2):

        super(RobertaForSeqCls, self).__init__(token_makers)

        self.bert = True  # for optimizer's model parameters

        self.num_classes = num_classes

        self._model = RobertaModel.from_pretrained(
            pretrained_model_name, cache_dir=str(CachePath.ROOT)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self._model.config.hidden_size, self._model.config.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(self._model.config.hidden_size, num_classes)
        )
        self.classifier.apply(self._model.init_weights)

        self.criterion = nn.CrossEntropyLoss()

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
            - class_logits: representing unnormalized log probabilities of the class.

            - class_idx: target class idx
            - data_idx: data idx
            - loss: a scalar loss to be optimized
        """

        bert_inputs = features["bert_input"]["feature"]
        attention_mask = (bert_inputs > 0).long()

        outputs = self._model(
            bert_inputs, token_type_ids=None, attention_mask=attention_mask
        )
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]  # take <s> token (equiv. to [CLS])
        class_logits = self.classifier(pooled_output)

        output_dict = {"sequence_embed": pooled_output, "class_logits": class_logits}

        if labels:
            class_idx = labels["class_idx"]
            data_idx = labels["data_idx"]

            output_dict["class_idx"] = class_idx
            output_dict["data_idx"] = data_idx

            # Loss
            loss = self.criterion(
                class_logits.view(-1, self.num_classes), class_idx.view(-1)
            )
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

        data_idx = inputs["labels"]["data_idx"][index].item()
        data_id = self._dataset.get_id(data_idx)

        helper = self._dataset.helper

        sequence_a = helper["examples"][data_id]["sequence_a"]
        sequence_a_tokens = helper["examples"][data_id]["sequence_a_tokens"]
        sequence_b = helper["examples"][data_id]["sequence_b"]
        sequence_b_tokens = helper["examples"][data_id]["sequence_b_tokens"]
        target_class_text = helper["examples"][data_id]["class_text"]

        pred_class_idx = predictions[data_id]["class_idx"]
        pred_class_text = self._dataset.get_class_text_with_idx(pred_class_idx)

        print()
        print("- Sequence a:", sequence_a)
        print("- Sequence a Tokens:", sequence_a_tokens)
        if sequence_b:
            print("- Sequence b:", sequence_b)
            print("- Sequence b Tokens:", sequence_b_tokens)
        print("- Target:")
        print("    Class:", target_class_text)
        print("- Predict:")
        print("    Class:", pred_class_text)
        print()
