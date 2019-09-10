
from claf.data import utils


class BertFeature:
    """
    BertFeature Data Transfer Object (DTO) Class

    dictionary consisting of
        - bert_input: indexed bert_input feature
        - token_type: segment_ids feature
    """

    BERT_INPUT = "bert_input"
    TOKEN_TYPE = "token_type"  #segment_id

    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def set_input(self, bert_input):
        self.__dict__[self.BERT_INPUT] = bert_input
        self.set_feature(self.TOKEN_TYPE, utils.make_bert_token_type(bert_input))

    def set_input_with_speical_token(self, *args, **kwargs):
        bert_input = utils.make_bert_input(*args, **kwargs)
        self.set_input(bert_input)

    def set_feature(self, key, value):
        self.__dict__[key] = {"feature": value, "text": ""}

    def to_dict(self):
        return dict(self.__dict__)
