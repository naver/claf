
from collections import Counter
import json
import logging
import re

from overrides import overrides
from tqdm import tqdm

from claf.data import utils
from claf.data.dataset import SQuADBertDataset
from claf.data.batch import make_batch
from claf.data.reader.base import DataReader
from claf.decorator import register
from claf.metric.squad_v1_official import normalize_answer
from claf.tokens.tokenizer import SentTokenizer, WordTokenizer

logger = logging.getLogger(__name__)


class Token:
    def __init__(self, text, text_span=None):
        self.text = text
        self.text_span = text_span


@register("reader:squad_bert")
class SQuADBertReader(DataReader):
    """
    SQuAD DataReader for BERT

    * Args:
        file_paths: .json file paths (train and dev)
        tokenizers: defined tokenizers config (char/word)
    """

    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"

    def __init__(
        self,
        file_paths,
        lang_code,
        tokenizers,
        max_seq_length=384,
        context_stride=128,
        max_question_length=64,
    ):

        super(SQuADBertReader, self).__init__(file_paths, SQuADBertDataset)
        self.lang_code = lang_code
        self.max_seq_length = max_seq_length
        self.context_stride = context_stride
        self.max_question_length = max_question_length

        self.text_columns = ["bert_input", "context", "question"]

        if "subword" not in tokenizers:
            raise ValueError("WordTokenizer and SubwordTokenizer is required.")

        sent_tokenizer = SentTokenizer("punkt", {})
        if lang_code == "ko":
            self.word_tokenizer = WordTokenizer("mecab_ko", sent_tokenizer, split_with_regex=True)
        else:
            self.word_tokenizer = WordTokenizer(
                "treebank_en", sent_tokenizer, split_with_regex=True
            )
        self.subword_tokenizer = tokenizers["subword"]

    @overrides
    def _read(self, file_path, data_type=None):
        word_tokenized_error_count, subword_tokenized_error_count = 0, 0

        if data_type != "train":
            self.context_stride = 64  # NOTE: hard-code

        data = self.data_handler.read(file_path)
        squad = json.loads(data)
        if "data" in squad:
            squad = squad["data"]

        helper = {
            "file_path": file_path,
            "examples": {},
            "raw_dataset": squad,
            "cls_token": self.CLS_TOKEN,
            "sep_token": self.SEP_TOKEN,

            "model": {
                "lang_code": self.lang_code,
            },
        }
        features, labels = [], []

        for article in tqdm(squad, desc=data_type):
            for paragraph in article["paragraphs"]:
                context_text = paragraph["context"].replace("``", '" ').replace("''", '" ')
                context_tokens = self.word_tokenizer.tokenize(context_text)

                context_spans, char_to_word_offset = self._convert_to_spans(
                    context_text, context_tokens
                )
                context_tokens = [
                    Token(text, span) for (text, span) in zip(context_tokens, context_spans)
                ]

                context_sub_tokens = []
                for token in context_tokens:
                    for sub_token in self.subword_tokenizer.tokenize(token.text):
                        context_sub_tokens.append(Token(sub_token, token.text_span))

                for qa in paragraph["qas"]:
                    question_text = qa["question"]
                    question_text = " ".join(self.word_tokenizer.tokenize(question_text))
                    question_sub_tokens = [
                        Token(subword) for subword in self.subword_tokenizer.tokenize(question_text)
                    ]

                    id_ = qa["id"]
                    answers = qa["answers"]

                    answer_texts, answer_indices = [], []

                    if qa.get("is_impossible", None):
                        answers = qa["plausible_answers"]
                        answerable = 0
                    else:
                        answers = qa["answers"]
                        answerable = 1

                    for answer in answers:
                        answer_start = answer["answer_start"]
                        answer_end = answer_start + len(answer["text"]) - 1

                        answer_texts.append(answer["text"])
                        answer_indices.append((answer_start, answer_end))

                    if len(answer_indices) > 0:
                        answer_char_start, answer_char_end = self._find_one_most_common(
                            answer_indices
                        )
                        answer_word_start = char_to_word_offset[answer_char_start]
                        answer_word_end = char_to_word_offset[answer_char_end]

                        char_answer_text = context_text[answer_char_start : answer_char_end + 1]
                        word_answer_text = context_text[
                            context_spans[answer_word_start][0] : context_spans[answer_word_end][1]
                        ]

                        if not self._is_rebuild(char_answer_text, word_answer_text):
                            logger.warning(f"word_tokenized_error: {char_answer_text}  ###  {word_answer_text}")
                            word_tokenized_error_count += 1
                    else:
                        # Unanswerable
                        answers = ["<noanswer>"]
                        answer_char_start, answer_char_end = -1, -1
                        answer_word_start, answer_word_end = -1, -1

                    bert_features, bert_labels = self._make_features_and_labels(
                        context_sub_tokens,
                        question_sub_tokens,
                        answer_char_start,
                        answer_char_end + 1,
                    )

                    for (index, (feature, label)) in enumerate(zip(bert_features, bert_labels)):
                        bert_tokens = feature
                        answer_start, answer_end = label

                        if (
                            answer_start < 0
                            or answer_start >= len(bert_tokens)
                            or answer_end >= len(bert_tokens)
                            or bert_tokens[answer_start].text_span is None
                            or bert_tokens[answer_end].text_span is None
                        ):
                            continue

                        char_start = bert_tokens[answer_start].text_span[0]
                        char_end = bert_tokens[answer_end].text_span[1]
                        bert_answer = context_text[char_start:char_end]

                        if char_answer_text != bert_answer:
                            logger.warning(f"subword_tokenized_error: {char_answer_text} ### {word_answer_text})")
                            subword_tokenized_error_count += 1

                        feature_row = {
                            "bert_input": [token.text for token in bert_tokens],
                            "bert_token": bert_tokens,
                        }
                        features.append(feature_row)

                        bert_id = id_ + f"#{index}"
                        label_row = {
                            "id": bert_id,  # question_id + bert_index
                            "answer_texts": "\t".join(answer_texts),
                            "answer_start": answer_start,
                            "answer_end": answer_end,
                            "answerable": answerable,
                        }
                        labels.append(label_row)

                        if id_ not in helper["examples"]:
                            helper["examples"][id_] = {
                                "context": context_text,
                                "question": question_text,
                                "answers": answer_texts,
                            }
                        helper["examples"][id_][f"bert_tokens_{index}"] = bert_tokens

        logger.info(
            f"tokenized_error_count - word: {word_tokenized_error_count} | subword: {subword_tokenized_error_count}"
        )
        return make_batch(features, labels), helper

    @overrides
    def read_one_example(self, inputs):
        """ inputs keys: question, context """
        context_text = inputs["context"].replace("``", '" ').replace("''", '" ')
        tokenized_context = self.word_tokenizer.tokenize(context_text)
        context_spans, char_to_word_offset = self._convert_to_spans(context_text, tokenized_context)
        context_tokens = [
            Token(text, span) for (text, span) in zip(tokenized_context, context_spans)
        ]

        context_sub_tokens = []
        for token in context_tokens:
            for sub_token in self.subword_tokenizer.tokenize(token.text):
                context_sub_tokens.append(Token(sub_token, token.text_span))

        question_text = inputs["question"]
        question_text = " ".join(self.word_tokenizer.tokenize(question_text))
        question_sub_tokens = [
            Token(subword) for subword in self.subword_tokenizer.tokenize(question_text)
        ]

        bert_tokens, _ = self._make_features_and_labels(
            context_sub_tokens, question_sub_tokens, -1, -1
        )

        helper = {
            "bert_token": [],
            "tokenized_context": tokenized_context,
            "token_key": "tokenized_context"  # for 1-example inference latency key
        }

        features = []
        for bert_token in bert_tokens:
            bert_input = [token.text for token in bert_token]
            token_type = utils.make_bert_token_type(bert_input, SEP_token=self.SEP_TOKEN)

            features.append(
                {
                    "bert_input": bert_input,
                    "token_type": {"feature": token_type, "text": ""},  # TODO: fix hard-code
                }
            )
            helper["bert_token"].append(bert_token)
        return features, helper

    def _find_one_most_common(self, answers):
        answer_counter = Counter(answers)
        value = answer_counter.most_common(1)[0][0]
        return value[0], value[1]

    def _convert_to_spans(self, raw_text, tokenized_text):
        """ Convert a tokenized version of `raw_text` into a series character spans referencing the `raw_text` """
        double_quote_re = re.compile("\"|``|''")

        curr_idx = 0
        spans = []
        char_to_words = [-1 for _ in range(len(raw_text))]

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
                joined_tokenized_text = " ".join(tokenized_text)
                raise ValueError(
                    f"\n{raw_text} \n\n{joined_tokenized_text} \nToken: {token}, Index: {temp}, Current Index: {curr_idx}"
                )
            curr_idx = temp
            spans.append((curr_idx, curr_idx + token_length))
            curr_idx += token_length

            start, end = spans[-1]
            for i in range(start, end):
                char_to_words[i] = len(spans) - 1

        for i in range(len(raw_text)):
            if char_to_words[i] != -1:
                continue

            for j, span in enumerate(spans):
                start, end = span
                if start < i <= end:
                    char_to_words[i] = j

        return spans, char_to_words

    def _is_rebuild(self, char_answer_text, word_answer_text):
        norm_char_answer_text = normalize_answer(char_answer_text)
        norm_word_answer_text = normalize_answer(word_answer_text)

        if norm_char_answer_text != norm_word_answer_text:
            return False
        else:
            return True

    def _make_features_and_labels(
        self, context_sub_tokens, question_sub_tokens, answer_char_start, answer_char_end
    ):
        # subword, context_stride logic with context_max_length
        context_max_length = (
            self.max_seq_length - len(question_sub_tokens) - 3
        )  # [CLS], [SEP], [SEP]
        start_offset = 0

        context_stride_spans = []
        while start_offset < len(context_sub_tokens):
            strided_context_length = len(context_sub_tokens) - start_offset
            if strided_context_length > context_max_length:
                strided_context_length = context_max_length

            context_stride_spans.append((start_offset, strided_context_length))
            if start_offset + strided_context_length == len(context_sub_tokens):
                break
            start_offset += min(strided_context_length, self.context_stride)

        features, labels = [], []
        for (start_offset, length) in context_stride_spans:
            bert_tokens = [Token(self.CLS_TOKEN)]
            bert_tokens += question_sub_tokens[: self.max_question_length]
            bert_tokens += [Token(self.SEP_TOKEN)]
            bert_tokens += context_sub_tokens[start_offset : start_offset + length]
            bert_tokens += [Token(self.SEP_TOKEN)]
            features.append(bert_tokens)

            if answer_char_start == -1 and answer_char_end == -1:
                answer_start, answer_end = 0, 0
            else:
                answer_start, answer_end = self._get_closest_answer_spans(
                    bert_tokens, answer_char_start, answer_char_end
                )

            labels.append((answer_start, answer_end))
        return features, labels

    def _get_closest_answer_spans(self, tokens, char_start, char_end):
        NONE_VALUE, DISTANCE_THRESHOLD = -100, 2

        text_spans = [
            (NONE_VALUE, NONE_VALUE) if token.text_span is None else token.text_span
            for token in tokens
        ]

        start_distances = [abs(span[0] - char_start) for span in text_spans]
        end_distances = [abs(span[1] - char_end) for span in text_spans]

        min_start_distance, min_end_distance = min(start_distances), min(end_distances)
        if min_start_distance < DISTANCE_THRESHOLD:
            answer_start = start_distances.index(min_start_distance)
        else:
            answer_start = 0

        if min_end_distance < DISTANCE_THRESHOLD:
            answer_end = end_distances.index(min_end_distance)
            start_from = answer_end + 1
            try:
                # e.g.) end_distances: [3, 1, 1, 4], min_end_distance = 1 => use 2 index instead of 1
                answer_end = end_distances.index(min_end_distance, start_from)
            except ValueError:
                pass
        else:
            answer_end = 0
        return answer_start, answer_end
