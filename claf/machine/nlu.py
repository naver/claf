
import logging
import json

from overrides import overrides

from claf.data.data_handler import CachePath, DataHandler
from claf.decorator import register

from claf.machine.base import Machine
import claf.utils as common_utils


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
    def __call__(self, input_):
        if isinstance(input_, str):
            utterance = input_
            return_logits = False

        elif isinstance(input_, dict):
            error_message = """
            Invalid input data format
            
            Proper data format: {
                "input": {
                    "sequence": <str>
                }
                "arguments": {
                    "return_logits": <bool>
                }
            }
            """

            try:
                utterance = input_["input"]["sequence"]
                return_logits = input_["arguments"]["return_logits"]
            except:
                return json.dumps({
                    "error": error_message,
                    "output": None,
                })

        else:
            return json.dumps({
                "error": "invalid input type",
                "output": None,
            })

        result_dict = {
            "error": None,
            "output": {
                "intent": self.intent_classification(utterance, return_logits),
                "slots": self.slot_filling(utterance, return_logits),
            }
        }
        return json.dumps(common_utils.serializable(result_dict), ensure_ascii=False)

    def intent_classification(self, utterance, return_logits=False):
        raw_feature = {
            "sequence": utterance,
            "return_logits": return_logits,
        }
        return self.ic_experiment.predict(raw_feature)

    def slot_filling(self, utterance, return_logits=False):
        raw_feature = {
            "sequence": utterance,
            "return_logits": return_logits,
        }
        return self.sf_experiment.predict(raw_feature)
