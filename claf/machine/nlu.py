
import logging

from overrides import overrides

from claf.data.data_handler import CachePath, DataHandler
from claf.decorator import register

from claf.machine.base import Machine


logger = logging.getLogger(__name__)


@register("machine:nlu")
class NLU(Machine):
    """
    Natural Language Understanding Machine

    * Args:
        config: machine_config
    """

    def __init__(self, config):
        super(NLU, self).__init__(config)
        self.data_handler = DataHandler(CachePath.MACHINE / "nlu")

        self.load()

    @overrides
    def load(self):
        # NLU
        # - Intent Classification Experiment
        # - Slot Filling Experiment

        nlu_config = self.config.nlu

        self.ic_experiment = self.make_module(nlu_config.intent)
        self.sf_experiment = self.make_module(nlu_config.slots)
        print("Ready ..! \n")

    @overrides
    def __call__(self, utterance):

        nlu_result = dict()

        intent_info = self.intent_classification(utterance)
        nlu_result.update({"intent": intent_info["class_text"]})

        slots_info = self.slot_filling(utterance)
        nlu_result.update({"slots": slots_info["tag_dict"]})

        return nlu_result

    def intent_classification(self, utterance):
        raw_feature = {"sequence": utterance}
        return self.ic_experiment.predict(raw_feature)

    def slot_filling(self, utterance):
        raw_feature = {"sequence": utterance}
        return self.sf_experiment.predict(raw_feature)
