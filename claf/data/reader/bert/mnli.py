
import logging

from overrides import overrides

from claf.data.reader import SeqClsBertReader
from claf.decorator import register

logger = logging.getLogger(__name__)


@register("reader:mnli_bert")
class MNLIBertReader(SeqClsBertReader):
    """
    MNLI DataReader for BERT

    * Args:
        file_paths: .tsv file paths (train and dev)
        tokenizers: defined tokenizers config
    """

    CLASS_DATA = ["contradiction", "entailment", "neutral"]

    def __init__(
        self,
        file_paths,
        tokenizers,
        sequence_max_length=None,
        cls_token="[CLS]",
        sep_token="[SEP]",
        input_type="bert",
        is_test=False,
    ):

        super(MNLIBertReader, self).__init__(
            file_paths,
            tokenizers,
            sequence_max_length,
            class_key=None,
            cls_token=cls_token,
            sep_token=sep_token,
            input_type=input_type,
            is_test=is_test,
        )

    @overrides
    def _get_data(self, file_path, **kwargs):
        data_type = kwargs["data_type"]

        _file = self.data_handler.read(file_path)
        lines = _file.split("\n")

        data = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            line_tokens = line.split("\t")
            if len(line_tokens) <= 1:
                continue
            data.append({
                "uid": f"{data_type}-{i}",
                "sequence_a": line_tokens[8],
                "sequence_b": line_tokens[9],
                self.class_key: str(line_tokens[-1]),
            })

        return data
