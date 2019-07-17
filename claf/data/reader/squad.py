
from collections import Counter
import json
import logging
import re

from overrides import overrides
from tqdm import tqdm

from claf.data.dataset import SQuADDataset
from claf.data.batch import make_batch
from claf.data.reader.base import DataReader
from claf.decorator import register
from claf.metric.squad_v1_official import normalize_answer

logger = logging.getLogger(__name__)


@register("reader:squad")
class SQuADReader(DataReader):
    """
    SQuAD DataReader

    * Args:
        file_paths: .json file paths (train and dev)
        tokenizers: defined tokenizers config (char/word)
    """

    def __init__(self, file_paths, lang_code, tokenizers, context_max_length=None):
        super(SQuADReader, self).__init__(file_paths, SQuADDataset)
        self.lang_code = lang_code
        self.context_max_length = context_max_length

        self.text_columns = ["context", "question"]

        if "word" not in tokenizers:
            raise ValueError("WordTokenizer is required. define English WordTokenizer")
        self.word_tokenizer = tokenizers["word"]

    @overrides
    def _read(self, file_path, data_type=None):
        tokenized_error_count = 0

        data = self.data_handler.read(file_path)
        squad = json.loads(data)
        if "data" in squad:
            squad = squad["data"]

        helper = {
            "file_path": file_path,
            "examples": {},  # qid: {context: ..., text_span: ..., question: ..., answer_texts}
            "raw_dataset": squad,

            "model": {
                "lang_code": self.lang_code,
            },
        }

        features, labels = [], []

        for article in tqdm(squad, desc=data_type):
            for paragraph in article["paragraphs"]:
                context = paragraph["context"].replace("``", '" ').replace("''", '" ')
                context_words = self.word_tokenizer.tokenize(context)

                if (
                    self.context_max_length is not None
                    and data_type == "train"
                    and len(context_words) > self.context_max_length
                ):
                    continue

                for qa in paragraph["qas"]:
                    question = qa["question"].strip().replace("\n", "")
                    id_ = qa["id"]

                    answer_texts, answer_indices = [], []

                    if qa.get("is_impossible", None):
                        answers = qa["plausible_answers"]
                        answerable = 0
                    else:
                        answers = qa["answers"]
                        answerable = 1

                    for answer in answers:
                        answer_start = answer["answer_start"]
                        answer_end = answer_start + len(answer["text"])

                        answer_texts.append(answer["text"])
                        answer_indices.append((answer_start, answer_end))

                    feature_row = {
                        "context": self._clean_text(context),
                        "question": question,
                    }
                    features.append(feature_row)

                    if len(answer_indices) > 0:
                        answer_start, answer_end = self._find_one_most_common(answer_indices)
                        text_spans = self._convert_to_spans(context, context_words)
                        word_idxs = self._get_word_span_idxs(text_spans, answer_start, answer_end)

                        word_answer_start = word_idxs[0]
                        word_answer_end = word_idxs[-1]

                        # To check rebuild answer: char_answer_text - word_answer_text
                        char_answer_text = context[answer_start:answer_end]
                        word_answer_text = context[
                            text_spans[word_answer_start][0] : text_spans[word_answer_end][1]
                        ]

                        if not self._is_rebuild(char_answer_text, word_answer_text):
                            logger.warning(f"word_tokenized_error: {char_answer_text}  ###  {word_answer_text}")
                            tokenized_error_count += 1

                    else:
                        # Unanswerable
                        answers = ["<noanswer>"]
                        text_spans = []
                        answer_start, answer_end = 0, 0
                        word_answer_start, word_answer_end = 0, 0

                    label_row = {
                        "id": id_,
                        "answer_start": word_answer_start,
                        "answer_end": word_answer_end,
                        "answerable": answerable,
                    }
                    labels.append(label_row)

                    helper["examples"][id_] = {
                        "context": context,
                        "text_span": text_spans,
                        "question": question,
                        "answers": answer_texts,
                    }

        logger.info(f"tokenized_error_count: {tokenized_error_count} ")
        return make_batch(features, labels), helper

    @overrides
    def read_one_example(self, inputs):
        """ inputs keys: question, context """
        context_text = inputs["context"]
        tokenized_context = self.word_tokenizer.tokenize(context_text)
        question_text = inputs["question"].strip().replace("\n", "")

        features = {}
        features["context"] = self._clean_text(context_text)
        features["question"] = self._clean_text(question_text)

        helper = {
            "text_span": self._convert_to_spans(context_text, tokenized_context),
            "tokenized_context": tokenized_context,
            "token_key": "tokenized_context"  # for 1-example inference latency key
        }
        return features, helper

    def _clean_text(self, text):
        # https://github.com/allenai/document-qa/blob/2f9fa6878b60ed8a8a31bcf03f802cde292fe48b/docqa/data_processing/text_utils.py#L124
        # be consistent with quotes, and replace \u2014 and \u2212 which I have seen being mapped to UNK
        # by glove word vecs
        return (
            text.replace("''", '"')
            .replace("``", '"')
            .replace("\u2212", "-")
            .replace("\u2014", "\u2013")
        )

    def _find_one_most_common(self, answers):
        answer_counter = Counter(answers)
        value = answer_counter.most_common(1)[0][0]
        return value[0], value[1]

    def _convert_to_spans(self, raw_text, tokenized_text):
        """ Convert a tokenized version of `raw_text` into a series character spans referencing the `raw_text` """
        double_quote_re = re.compile("\"|``|''")

        curr_idx = 0
        spans = []
        for token in tokenized_text:
            # Tokenizer might transform double quotes, for this case search over several
            # possible encodings
            if double_quote_re.match(token):
                span = double_quote_re.search(raw_text[curr_idx:])
                temp = curr_idx + span.start()
                token_length = span.end() - span.start()
            else:
                temp = raw_text.find(token, curr_idx)
                token_length = len(token)
            if temp < curr_idx:
                raise ValueError(f"{raw_text} \n{tokenized_text} \n{token}")
            curr_idx = temp
            spans.append((curr_idx, curr_idx + token_length))
            curr_idx += token_length
        return spans

    def _get_word_span_idxs(self, spans, start, end):
        idxs = []
        for word_ix, (s, e) in enumerate(spans):
            if e > start:
                if s < end:
                    idxs.append(word_ix)
                else:
                    break
        return idxs

    def _is_rebuild(self, char_answer_text, word_answer_text):
        norm_char_answer_text = normalize_answer(char_answer_text)
        norm_word_answer_text = normalize_answer(word_answer_text)

        if norm_char_answer_text != norm_word_answer_text:
            return False
        else:
            return True
