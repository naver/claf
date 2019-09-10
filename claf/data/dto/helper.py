


class Helper:
    """
    Helper Data Transfer Object (DTO) Class
      (include model parameter - value defined by data, predict_helper and etc.)

    dictionary consisting of
        - model: (dict) model parameter (ex. num_classes)
        - predict_helper: (dict) predict_helper (ex. class_idx2text)

    """

    EXAMPLES = "examples"
    MODEL = "model"
    PREDICT_HELPER = "predict_helper"

    def __init__(self, **kwargs):
        self.__dict__ = kwargs

        default_keys = [self.EXAMPLES, self.MODEL, self.PREDICT_HELPER]
        for key in default_keys:
            if key not in self.__dict__:
                self.__dict__[key] = {}

    def set_example(self, uid, example, update=False):
        if update:
            self.__dict__[self.EXAMPLES][uid].update(example)
        else:
            self.__dict__[self.EXAMPLES][uid] = example

    def set_model_parameter(self, parameters):
        self.__dict__[self.MODEL] = parameters

    def set_predict_helper(self, predict_helper):
        self.__dict__[self.PREDICT_HELPER] = predict_helper

    def to_dict(self):
        return dict(self.__dict__)
