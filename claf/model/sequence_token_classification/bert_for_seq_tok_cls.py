
from overrides import overrides
from pytorch_pretrained_bert.modeling import BertModel
import torch.nn as nn

from claf.data.data_handler import CachePath
from claf.decorator import register
from claf.model.base import ModelWithoutTokenEmbedder
from claf.model.sequence_token_classification.mixin import SequenceTokenClassification

from claf.model import cls_utils
from claf.model.token_classification.crf import ConditionalRandomField, allowed_transitions
from claf.modules.criterion import get_criterion_fn
from claf.config.namespace import NestedNamespace


@register("model:bert_for_seq_tok_cls")
class BertForSeqTokCls(SequenceTokenClassification, ModelWithoutTokenEmbedder):
    """
    Implementation of Joint version of Sentence Classification and Tagging model presented in
    BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
    (https://arxiv.org/abs/1810.04805)

    * Args:
        token_embedder: used to embed the sequence
        num_classes: number of classified classes
        num_tags: number of classified tags
        tag_idx2text: dictionary mapping from tag index to text
        ignore_tag_idx: index of the tag to ignore when calculating loss (tag pad value)
        class_criterion: class criterion function config
        tag_criterion: tag criterion function config

    * Kwargs:
        pretrained_model_name: the name of a pre-trained model
        class_dropout: sequence classification layer dropout
        tag_dropout: tag classification layer dropout
        K: K for top K accuracy
    """

    def __init__(
        self,
        token_makers,
        num_classes,
        num_tags,
        tag_idx2text,
        ignore_tag_idx,
        class_criterion,
        tag_criterion,
        pretrained_model_name=None,
        class_dropout=0.2,
        tag_dropout=0.2,
        K=2,
    ):

        super(BertForSeqTokCls, self).__init__(token_makers)

        self.bert = True  # for optimizer's model parameters

        self.num_classes = num_classes

        self.ignore_tag_idx = ignore_tag_idx
        self.num_tags = num_tags

        self._model = BertModel.from_pretrained(
            pretrained_model_name, cache_dir=str(CachePath.ROOT)
        )

        self.class_classifier = nn.Sequential(
            nn.Dropout(class_dropout), nn.Linear(self._model.config.hidden_size, num_classes)
        )
        self.class_classifier.apply(self._model.init_bert_weights)

        self.tag_classifier = nn.Sequential(
            nn.Dropout(tag_dropout), nn.Linear(self._model.config.hidden_size, num_tags)
        )
        self.tag_classifier.apply(self._model.init_bert_weights)

        self.tag_loss_weight = getattr(tag_criterion, "weight", 0.5)
        self.class_loss_weight = getattr(class_criterion, "weight", 0.5)

        self.class_criterion = get_criterion_fn(
            class_criterion.name,
            **vars(getattr(class_criterion, class_criterion.name, NestedNamespace()))
        )

        self.use_crf = "crf" in tag_criterion.name
        if self.use_crf:
            self.crf = ConditionalRandomField(
                num_tags,
                allowed_transitions("BIO", tag_idx2text),
                include_start_end_transitions=False,
            )
            assert tag_criterion.name == "crf_negative_log_likelihood"
        else:
            self.crf = None

        tag_criterion_params = vars(getattr(tag_criterion, tag_criterion.name, NestedNamespace()))
        tag_criterion_params.update({"ignore_index": self.ignore_tag_idx, "crf": self.crf})
        self.tag_criterion = get_criterion_fn(tag_criterion.name, **tag_criterion_params)

        self.K = K

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
                "class_idx": [2, 1, 0, 4, 5, ...],
                "tag_idxs": [7, 3, 4, 2, 5, ...],
                "data_idx": [2, 4, 5, 7, 2, 1, ...]
            }
            Do not calculate loss when there is no label. (inference/predict mode)

        * Returns: output_dict (dict) consisting of
            - sequence_embed: embedding vector of the sequence
            - class_logits: representing unnormalized log probabilities of the class.
            - tag_logits: representing unnormalized log probabilities of the tags.

            - class_idx: target class idx
            - tag_idxs: target class idx
            - data_idx: data idx
            - class_loss: class loss
            - tag_loss: tag loss
            - loss: a scalar loss to be optimized, which is weighted sum of class_loss and tag_loss
        """

        bert_inputs = features["bert_input"]["feature"]
        token_type_ids = features["token_type"]["feature"]
        tagged_sub_token_idxs = features["tagged_sub_token_idxs"]["feature"]
        num_tokens = features["num_tokens"]["feature"]

        attention_mask = (bert_inputs > 0).long()

        token_encodings, sequence_embed = self._model(
            bert_inputs, token_type_ids, attention_mask, output_all_encoded_layers=False
        )
        class_logits = self.class_classifier(sequence_embed)
        tag_logits = self.tag_classifier(token_encodings)  # [B, L, num_tags]

        # gather the logits of the tagged token positions.
        gather_token_pos_idxs = tagged_sub_token_idxs.unsqueeze(-1).repeat(1, 1, self.num_tags)
        token_tag_logits = tag_logits.gather(1, gather_token_pos_idxs)  # [B, num_tokens, num_tags]

        sliced_token_tag_logits = [token_tag_logits[idx, :n, :] for idx, n in enumerate(num_tokens)]

        output_dict = {
            "sequence_embed": sequence_embed,
            "class_logits": class_logits,
            "tag_logits": sliced_token_tag_logits,
        }

        if self.use_crf:
            mask = (tagged_sub_token_idxs > 0).long()
            best_paths = self.crf.viterbi_tags(token_tag_logits, mask)
            predicted_tags = [x for x, y in best_paths]
            output_dict["pred_tag_idxs"] = predicted_tags

        if labels:
            class_idx = labels["class_idx"]
            tag_idxs = labels["tag_idxs"]
            data_idx = labels["data_idx"]

            output_dict["class_idx"] = class_idx
            output_dict["tag_idxs"] = tag_idxs
            output_dict["data_idx"] = data_idx

            # Loss
            class_loss = self.class_criterion(class_logits, class_idx)
            tag_loss = self.tag_criterion(token_tag_logits, tag_idxs)

            loss = self.class_loss_weight * class_loss + self.tag_loss_weight * tag_loss

            output_dict["class_loss"] = class_loss.unsqueeze(0)
            output_dict["tag_loss"] = tag_loss.unsqueeze(0)
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
                    - tag_idxs

        * Returns:
            print(Sequence, Sequence Tokens, Target Class, Target Tags, Target Entities, Predicted Class, Predicted Tags, Predicted Entities)
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

        sequence_tokens = helper["examples"][data_id]["sequence_sub_tokens"]

        print()
        print("- Sequence:", sequence)
        print("- Sequence Tokens:", sequence_tokens)
        print("- Target:")
        print("    Class:", target_class_text)
        print("    Tags:", target_tag_texts)
        print("    (Entities)", cls_utils.get_tag_entities(sequence, target_tag_texts))
        print("- Predict:")
        print("    Class:", pred_class_text)
        print("    Tags:", pred_tag_texts)
        print("    (Entities)", cls_utils.get_tag_entities(sequence, pred_tag_texts))
        print()
