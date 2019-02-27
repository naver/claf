
import json
from overrides import overrides

from claf.data import utils
from claf.data.collate import PadCollator
from claf.data.dataset.base import DatasetBase


class SQuADDataset(DatasetBase):
    """
    SQuAD Dataset
        compatible with v1.1 and v2.0

    * Args:
        batch: Batch DTO (claf.data.batch)

    * Kwargs:
        helper: helper from data_reader
    """

    def __init__(self, batch, helper=None):
        super(SQuADDataset, self).__init__()

        self.name = "squad"
        self.helper = helper
        self.raw_dataset = helper["raw_dataset"]  # for SQuAD official metric

        # Features
        self.context_idx = [feature["context"] for feature in batch.features]
        self.question_idx = [feature["question"] for feature in batch.features]

        self.features = [self.context_idx, self.question_idx]  # for lazy_evaluation

        # Labels
        self.qids = {data_index: label["id"] for (data_index, label) in enumerate(batch.labels)}
        self.data_indices = list(self.qids.keys())

        self.answers = {
            label["id"]: (
                label["answerable"],
                (label["answer_start"], label["answer_end"]),
            )
            for label in batch.labels
        }
        self.answer_starts = [label["answer_start"] for label in batch.labels]
        self.answer_ends = [label["answer_end"] for label in batch.labels]
        self.answerables = [label["answerable"] for label in batch.labels]

    @overrides
    def collate_fn(self, cuda_device_id=None):
        """ collate: indexed features and labels -> tensor """
        collator = PadCollator(cuda_device_id=cuda_device_id)

        def make_tensor_fn(data):
            context_idxs, question_idxs, data_idxs, \
                answer_starts, answer_ends, answerables = zip(*data)

            features = {
                "context": utils.transpose(context_idxs, skip_keys=["text"]),
                "question": utils.transpose(question_idxs, skip_keys=["text"]),
            }
            labels = {
                "answer_idx": data_idxs,
                "answer_start_idx": answer_starts,
                "answer_end_idx": answer_ends,
                "answerable": answerables,
            }
            return collator(features, labels)

        return make_tensor_fn

    @overrides
    def __getitem__(self, index):
        self.lazy_evaluation(index)

        return (
            self.context_idx[index],
            self.question_idx[index],
            self.data_indices[index],
            self.answer_starts[index],
            self.answer_ends[index],
            self.answerables[index],
        )

    def __len__(self):
        return len(self.qids)

    def __repr__(self):
        dataset_properties = {
            "name": self.name,
            "total_count": self.__len__(),
            "HasAns_count": len([True for item in self.answerables if item == 1]),
            "NoAns_count": len([False for item in self.answerables if item == 0]),
            "context_maxlen": self.context_maxlen,
            "question_maxlen": self.question_maxlen,
        }
        return json.dumps(dataset_properties, indent=4)

    @property
    def context_maxlen(self):
        return self._get_feature_maxlen(self.context_idx)

    @property
    def question_maxlen(self):
        return self._get_feature_maxlen(self.question_idx)

    def get_qid(self, data_index):
        return self.qids[data_index]

    def get_context(self, data_index):
        qid = self.get_qid(data_index)
        return self.helper["examples"][qid]["context"]

    def get_text_span(self, data_index):
        qid = self.get_qid(data_index)
        return self.helper["examples"][qid]["text_span"]

    @overrides
    def get_ground_truths(self, data_index):
        qid = self.get_qid(data_index)
        answer_texts = self.helper["examples"][qid]["answers"]
        answerable, answer_span = self.answers[qid]
        return answer_texts, answerable, answer_span

    @overrides
    def get_predict(self, data_index, start, end):
        return self.get_text_with_index(data_index, start, end)

    def get_text_with_index(self, data_index, start, end):
        if data_index is None:
            raise ValueError("qid or text is required.")

        context_text = self.get_context(data_index)
        text_span = self.get_text_span(data_index)

        if start >= len(text_span) or end >= len(text_span):
            # No_Answer Case
            return "<noanswer>"

        char_start = text_span[start][0]
        char_end = text_span[end][1]
        if char_start > char_end or len(context_text) <= char_end:
            return ""
        return context_text[char_start:char_end]
