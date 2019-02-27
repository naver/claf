
from overrides import overrides
from pytorch_pretrained_bert.modeling import BertModel
import torch.nn as nn

from claf.data.data_handler import CachePath
from claf.decorator import register
from claf.model.base import ModelWithoutTokenEmbedder
from claf.model.token_classification.mixin import TokenClassification

from claf.model import cls_utils


@register("model:bert_for_tok_cls")
class BertForTokCls(TokenClassification, ModelWithoutTokenEmbedder):
    """
    Implementation of Single Sentence Tagging model presented in
    BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
    (https://arxiv.org/abs/1810.04805)

    * Args:
        token_embedder: used to embed the sequence
        num_tags: number of classified tags
        ignore_tag_idx: index of the tag to ignore when calculating loss (tag pad value)

    * Kwargs:
        pretrained_model_name: the name of a pre-trained model
        dropout: classification layer dropout
    """

    def __init__(
        self, token_makers, num_tags, ignore_tag_idx, pretrained_model_name=None, dropout=0.2
    ):

        super(BertForTokCls, self).__init__(token_makers)

        self.bert = True  # for optimizer's model parameters

        self.ignore_tag_idx = ignore_tag_idx
        self.num_tags = num_tags

        self._model = BertModel.from_pretrained(
            pretrained_model_name, cache_dir=str(CachePath.ROOT)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self._model.config.hidden_size, num_tags)
        )
        self.classifier.apply(self._model.init_bert_weights)

        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_tag_idx)

    @overrides
    def forward(self, features, labels=None):
        """
        * Args:
            features: feature dictionary like below.
            {
                "bert_input": {
                    "feature": [
                        [100, 576, 21, 45, 7, 91, 101, 0, 0, ...],
                        ...,
                    ]
                }
                "token_type": {
                    "feature": [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, ...],
                        ...,
                    ]
                },
                "tagged_sub_token_idxs": {
                    [
                        [1, 3, 4, 0, 0, 0, 0, 0, 0, ...],
                        ...,
                    ]
                }
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
            - tag_logits: representing unnormalized log probabilities of the tags.

            - tag_idxs: target class idx
            - data_idx: data idx
            - loss: a scalar loss to be optimized
        """

        bert_inputs = features["bert_input"]["feature"]
        token_type_ids = features["token_type"]["feature"]
        tagged_sub_token_idxs = features["tagged_sub_token_idxs"]["feature"]
        num_tokens = features["num_tokens"]["feature"]

        attention_mask = (bert_inputs > 0).long()

        token_encodings, sequence_embed = self._model(
            bert_inputs, token_type_ids, attention_mask, output_all_encoded_layers=False
        )
        tag_logits = self.classifier(token_encodings)  # [B, L, num_tags]

        # gather the logits of the tagged token positions.
        gather_token_pos_idxs = tagged_sub_token_idxs.unsqueeze(-1).repeat(1, 1, self.num_tags)
        token_tag_logits = tag_logits.gather(1, gather_token_pos_idxs)  # [B, num_tokens, num_tags]

        sliced_token_tag_logits = [token_tag_logits[idx, :n, :] for idx, n in enumerate(num_tokens)]

        output_dict = {"sequence_embed": sequence_embed, "tag_logits": sliced_token_tag_logits}

        if labels:
            tag_idxs = labels["tag_idxs"]
            data_idx = labels["data_idx"]

            output_dict["tag_idxs"] = tag_idxs
            output_dict["data_idx"] = data_idx

            # Loss
            loss = self.criterion(token_tag_logits.view(-1, self.num_tags), tag_idxs.view(-1))
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
            print(Sequence, Sequence Tokens, Target Tags, Target Slots, Predicted Tags, Predicted Slots)
        """

        data_idx = inputs["labels"]["data_idx"][index].item()
        data_id = self._dataset.get_id(data_idx)

        helper = self._dataset.helper
        sequence = helper["examples"][data_id]["sequence"]
        target_tag_texts = helper["examples"][data_id]["tag_texts"]

        pred_tag_idxs = predictions[data_id]["tag_idxs"]
        pred_tag_texts = self._dataset.get_tag_texts_with_idxs(pred_tag_idxs)

        sequence_tokens = helper["examples"][data_id]["sequence_sub_tokens"]

        print()
        print("- Sequence:", sequence)
        print("- Sequence Tokens:", sequence_tokens)
        print("- Target:")
        print("    Tags:", target_tag_texts)
        print("    (Slots)", cls_utils.get_tag_dict(sequence, target_tag_texts))
        print("- Predict:")
        print("    Tags:", pred_tag_texts)
        print("    (Slots)", cls_utils.get_tag_dict(sequence, pred_tag_texts))
        print()
