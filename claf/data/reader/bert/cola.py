
import logging

from overrides import overrides

from claf.data.reader import SeqClsBertReader
from claf.decorator import register

logger = logging.getLogger(__name__)


@register("reader:cola_bert")
class CoLABertReader(SeqClsBertReader):
    """
    CoLA DataReader for BERT

    * Args:
        file_paths: .tsv file paths (train and dev)
        tokenizers: defined tokenizers config
    """

    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    UNK_TOKEN = "[UNK]"
    CONTINUE_SYMBOL = "##"

    def __init__(
        self,
        file_paths,
        tokenizers,
        sequence_max_length=None,
    ):

        super(CoLABertReader, self).__init__(
            file_paths,
            tokenizers,
            sequence_max_length,
        )

    @overrides
    def _get_data(self, file_path, **kwargs):
        data_type = kwargs["data_type"]

        _file = self.data_handler.read(file_path)
        lines = _file.split("\n")

        if data_type == "train":
            lines.pop(0)

        data = []
        for i, line in enumerate(lines):
            line_tokens = line.split("\t")
            if len(line_tokens) > 1:
                data.append({
                    "uid": f"{data_type}-{i}",
                    "sequence": line_tokens[1] if data_type == "test" else line_tokens[3],
                    self.class_key: str(0) if data_type == "test" else str(line_tokens[1])
                })

        return data, data

    @overrides
    def _get_class_dicts(self, **kwargs):
        class_idx2text = {
            class_idx: str(class_idx)
            for class_idx in [0, 1]
        }
        class_text2idx = {class_text: class_idx for class_idx, class_text in class_idx2text.items()}

        return class_idx2text, class_text2idx
