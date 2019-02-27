
import logging
from itertools import chain

from overrides import overrides

from claf.data.reader import TokClsBertReader
from claf.decorator import register

logger = logging.getLogger(__name__)


@register("reader:conll2003_bert")
class CoNLL2003BertReader(TokClsBertReader):
    """
     CoNLL2003 for BERT

    * Args:
        file_paths: file paths (train and dev)

    * Kwargs:
        ignore_tag_idx: prediction results that have this number as ground-truth idx are ignored
    """

    def __init__(
            self,
            file_paths,
            tokenizers,
            sequence_max_length=None,
            ignore_tag_idx=-1,
    ):

        super(CoNLL2003BertReader, self).__init__(
            file_paths,
            tokenizers,
            lang_code=None,
            sequence_max_length=sequence_max_length,
            ignore_tag_idx=ignore_tag_idx,
        )

    @overrides
    def _get_data(self, file_path):
        _file = self.data_handler.read(file_path)
        texts = _file.split("\n\n")
        texts.pop(0)

        data = []
        for text in texts:
            tokens = text.split("\n")
            if len(tokens) > 1:
                example = list(zip(*[token.split() for token in tokens]))
                data.append({
                    "sequence": " ".join(example[0]),
                    self.tag_key: list(example[-1]),
                })

        return data, data

    @overrides
    def _get_tag_dicts(self, **kwargs):
        data = kwargs["data"]
        tags = sorted(list(set(chain.from_iterable(d[self.tag_key] for d in data))))

        tag_idx2text = {tag_idx: tag_text for tag_idx, tag_text in enumerate(tags)}
        tag_text2idx = {tag_text: tag_idx for tag_idx, tag_text in tag_idx2text.items()}

        return tag_idx2text, tag_text2idx
