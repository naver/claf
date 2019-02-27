
import json
from overrides import overrides

from claf.data import utils
from claf.data.collate import PadCollator
from claf.data.dataset.base import DatasetBase


class SQuADBertDataset(DatasetBase):
    """
    SQuAD Dataset for BERT
        compatible with v1.1 and v2.0

    * Args:
        batch: Batch DTO (claf.data.batch)

    * Kwargs:
        helper: helper from data_reader
    """

    def __init__(self, batch, helper=None):
        super(SQuADBertDataset, self).__init__()

        self.name = "squad_bert"
        self.helper = helper
        self.raw_dataset = helper["raw_dataset"]

        # Features
        self.bert_input_idx = [feature["bert_input"] for feature in batch.features]
        SEP_token = self.helper.get("sep_token", "[SEP]")
        self.token_type_idx = utils.make_bert_token_types(self.bert_input_idx, SEP_token=SEP_token)

        self.features = [self.bert_input_idx, self.token_type_idx]  # for lazy_evaluation

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
            bert_input_idxs, token_type_idxs, data_idxs, answer_starts, answer_ends, answerables = zip(
                *data
            )

            features = {
                "bert_input": utils.transpose(bert_input_idxs, skip_keys=["text"]),
                "token_type": utils.transpose(token_type_idxs, skip_keys=["text"]),
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
            self.bert_input_idx[index],
            self.token_type_idx[index],
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
            "HasAns_count": len([True for k, v in self.answers.items() if v[1] == 1]),
            "NoAns_count": len([False for k, v in self.answers.items() if v[1] == 0]),
            "bert_input_maxlen": self.bert_input_maxlen,
        }
        return json.dumps(dataset_properties, indent=4)

    @property
    def bert_input_maxlen(self):
        return self._get_feature_maxlen(self.bert_input_idx)

    def get_qid(self, data_index):
        qid = self.qids[data_index]
        if "#" in qid:
            qid = qid.split("#")[0]
        return qid

    def get_qid_index(self, data_index):
        qid = self.qids[data_index]
        if "#" in qid:
            return qid.split("#")[1]
        return None

    def get_context(self, data_index):
        qid = self.get_qid(data_index)
        return self.helper["examples"][qid]["context"]

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
            raise ValueError("data_id or text is required.")

        context_text = self.get_context(data_index)
        bert_token = self.get_bert_tokens(data_index)

        if (
            start <= 0
            or end >= len(bert_token)
            or bert_token[start].text_span is None
            or bert_token[end].text_span is None
        ):
            # No_Answer Case
            return "<noanswer>"

        char_start = bert_token[start].text_span[0]
        char_end = bert_token[end].text_span[1]
        if char_start > char_end or len(context_text) <= char_end:
            return ""
        return context_text[char_start:char_end]

    def get_bert_tokens(self, data_index):
        qid = self.get_qid(data_index)
        index = self.get_qid_index(data_index)

        if index is None:
            raise ValueError("bert_qid must have 'bert_index' (bert_id: qid#bert_index)")

        bert_index = f"bert_tokens_{index}"
        return self.helper["examples"][qid][bert_index]
