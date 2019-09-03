
import logging

from overrides import overrides

from claf.data.reader import RegressionBertReader
from claf.decorator import register

logger = logging.getLogger(__name__)


@register("reader:stsb_bert")
class STSBBertReader(RegressionBertReader):
    """
    STS-B (Semantic Textual Similarity Benchmark) DataReader for BERT

    * Args:
        file_paths: .tsv file paths (train and dev)
        tokenizers: defined tokenizers config
    """

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

        super(STSBBertReader, self).__init__(
            file_paths,
            tokenizers,
            sequence_max_length,
            label_key="score",
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
                "sequence_a": line_tokens[7],
                "sequence_b": line_tokens[8],
                "score": float(line_tokens[-1]),
            })

        return data
