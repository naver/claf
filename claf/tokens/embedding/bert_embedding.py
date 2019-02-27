
from overrides import overrides

from pytorch_pretrained_bert.modeling import BertModel

import claf.modules.functional as f

from .base import TokenEmbedding


class BertEmbedding(TokenEmbedding):
    """
    BERT Embedding(Encoder)

    BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
    (https://arxiv.org/abs/1810.04805)

    * Args:
        vocab: Vocab (claf.tokens.vocab)

    * Kwargs:
        pretrained_model_name: ...
        use_as_embedding: ...
        trainable: Finetune or fixed
    """

    def __init__(self, vocab, pretrained_model_name=None, trainable=False, unit="subword"):
        super(BertEmbedding, self).__init__(vocab)
        self.trainable = trainable

        self.pad_index = vocab.get_index(vocab.pad_token)
        self.sep_index = vocab.get_index(vocab.sep_token)

        if unit != "subword":
            raise NotImplementedError("BertEmbedding is only available 'subword' unit, right now.")

        self.bert_model = BertModel.from_pretrained(pretrained_model_name)  # BertModel with config

    @overrides
    def forward(self, inputs):
        if inputs.size(1) > self.bert_model.config.max_position_embeddings:
            raise ValueError(
                f"max_seq_length in this bert_model is '{self.bert_model.config.max_position_embeddings}'. (input seq_length: {inputs.size(1)})"
            )

        # TODO: add text_unit option
        # current: sub_word (default) / later: sub_words --(average)--> word
        attention_mask = (inputs != self.pad_index).long()
        sequence_output, pooled_output = self.bert_model(
            inputs, attention_mask=attention_mask, output_all_encoded_layers=False
        )
        sequence_output = f.masked_zero(sequence_output, attention_mask)

        if not self.trainable:
            sequence_output = sequence_output.detach()
            pooled_output = pooled_output.detach()

        sequence_output = self.remove_cls_sep_token(inputs, sequence_output)
        return sequence_output

    @overrides
    def get_output_dim(self):
        return self.bert_model.config.hidden_size

    def remove_cls_sep_token(self, inputs, outputs):
        seq_mask = inputs.eq(self.sep_index).eq(0)
        outputs = f.masked_zero(outputs, seq_mask)
        return outputs[:, 1:-1, :]  # B, S_L, D
