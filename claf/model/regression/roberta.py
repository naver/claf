
from overrides import overrides
from pytorch_transformers import RobertaModel
import torch.nn as nn

from claf.data.data_handler import CachePath
from claf.decorator import register
from claf.model.base import ModelWithoutTokenEmbedder
from claf.model.regression.mixin import Regression


@register("model:roberta_for_reg")
class RobertaForRegression(Regression, ModelWithoutTokenEmbedder):
    """
    Implementation of Sentence Regression model presented in
    BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
    (https://arxiv.org/abs/1810.04805)

    * Args:
        token_makers: used to convert the sequence to feature

    * Kwargs:
        pretrained_model_name: the name of a pre-trained model
        dropout: classification layer dropout
    """

    def __init__(self, token_makers, pretrained_model_name=None, dropout=0.2):

        super(RobertaForRegression, self).__init__(token_makers)

        self.bert = True  # for optimizer's model parameters

        NUM_CLASSES = 1

        self._model = RobertaModel.from_pretrained(
            pretrained_model_name, cache_dir=str(CachePath.ROOT)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self._model.config.hidden_size, NUM_CLASSES)
        )
        self.classifier.apply(self._model.init_weights)

        self.criterion = nn.MSELoss()

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
                "score": [2, 1, 0, 4, 5, ...]
                "data_idx": [2, 4, 5, 7, 2, 1, ...]
            }
            Do not calculate loss when there is no labels. (inference/predict mode)

        * Returns: output_dict (dict) consisting of
            - sequence_embed: embedding vector of the sequence
            - logits: model's score

            - data_idx: data idx
            - score: target score
            - loss: a scalar loss to be optimized
        """

        bert_inputs = features["bert_input"]["feature"]
        attention_mask = (bert_inputs > 0).long()

        outputs = self._model(
            bert_inputs, token_type_ids=None, attention_mask=attention_mask
        )
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)

        output_dict = {"sequence_embed": pooled_output, "logits": logits}

        if labels:
            data_idx = labels["data_idx"]
            score = labels["score"]

            output_dict["data_idx"] = data_idx
            output_dict["score"] = score

            # Loss
            loss = self.criterion(logits.view(-1, 1), score.view(-1, 1).float())
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

        target_score = helper["examples"][data_id]["score"]
        pred_score = predictions[data_id]["score"]

        print()
        print("- Sequence a:", sequence_a)
        print("- Sequence a Tokens:", sequence_a_tokens)
        if sequence_b:
            print("- Sequence b:", sequence_b)
            print("- Sequence b Tokens:", sequence_b_tokens)
        print("- Target:")
        print("    Score:", target_score)
        print("- Predict:")
        print("    Score:", pred_score)
        print()
