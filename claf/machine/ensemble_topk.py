
from functools import reduce  # Valid in Python 2.6+, required in Python 3
import logging
import json
import operator

from overrides import overrides
from tqdm import tqdm

from claf.data.data_handler import CachePath, DataHandler
from claf.decorator import register
from claf.machine.base import Machine
from claf.metric.korquad_v1_official import evaluate, metric_max_over_ground_truths, f1_score, normalize_answer


logger = logging.getLogger(__name__)


@register("machine:mrc_ensemble")
class MRCEnsemble(Machine):
    """
    Machine Reading Comprehension Ensemble

    * Args:
        config: machine_config
    """

    def __init__(self, config):
        super(MRCEnsemble, self).__init__(config)
        self.data_handler = DataHandler(CachePath.MACHINE / "mrc_ensemble")

        self.load()

    @overrides
    def load(self):
        mrc_config = self.config.reading_comprehension

        # Model 1 - BERT-Kor
        self.rc_experiment1 = self.make_module(mrc_config.model_1)
        print("BERT-Kor ready ..! \n")

        # # Model 2 - BERT-Multilingual
        # self.rc_experiment2 = self.make_module(mrc_config.model_2)
        # print("BERT-Multilingual ready ..! \n")

        # # Model 3 - DocQA
        # self.rc_experiment3 = self.make_module(mrc_config.model_3)
        # print("DocQA ready ..! \n")

        # # Model 4 - DrQA
        # self.rc_experiment4 = self.make_module(mrc_config.model_4)
        # print("DrQA ready ..! \n")

        print("All ready ..! \n")

    def evaluate(self, file_path, output_path):
        # KorQuAD dataset...

        # def get_answer_after_clustering(predictions):
            # categories = {}

            # for l1 in predictions:
                # l1_text = l1["text"]
                # l1_text_normalized = normalize_answer(l1_text)

                # categories[l1_text] = {
                    # "items": [],
                    # "score": 0
                # }

                # for l2 in predictions:
                    # l2_text = l2["text"]
                    # l2_text_normalized = normalize_answer(l2_text)

                    # if l1_text_normalized in l2_text_normalized:
                        # categories[l1_text]["items"].append(l2)
                        # categories[l1_text]["score"] += l2["score"]

            # # # count items then score * 1.n
            # # for k, v in categories.items():
                # # ratio = 1 + (len(v["items"]) / 10)
                # # v["score"] *= ratio

            # highest_category = [categories[c] for c in sorted(categories, key=lambda x: categories[x]["score"], reverse=True)][0]
            # answer_text = sorted(highest_category["items"], key=lambda x: x["score"], reverse=True)[0]["text"]
            # return answer_text

        # def get_answer_after_clustering_marginal(predictions):
            # categories = {}

            # for l1 in predictions:
                # l1_text = l1["text"]
                # l1_text_normalized = normalize_answer(l1_text)

                # categories[l1_text] = {
                    # "items": [],
                    # "score": 0
                # }

                # for l2 in predictions:
                    # l2_text = l2["text"]
                    # l2_text_normalized = normalize_answer(l2_text)

                    # if l1_text_normalized in l2_text_normalized:
                        # categories[l1_text]["items"].append(l2)
                        # categories[l1_text]["score"] *= l2["score"]
                    # else:
                        # categories[l1_text]["score"] *= 0.01  # Default value

            # # count items then score * 1.n
            # for k, v in categories.items():
                # ratio = 1 + (len(v["items"]) / 10)
                # v["score"] *= ratio

            # highest_category = [categories[c] for c in sorted(categories, key=lambda x: categories[x]["score"], reverse=True)][0]
            # answer_text = sorted(highest_category["items"], key=lambda x: x["score"], reverse=True)[0]["text"]
            # return answer_text

        # def post_processing(text):
            # # detach josa
            # # josas = ['은', '는', '이', '가', '을', '를', '과', '와', '이다', '다', '으로', '로', '의', '에']
            # josas = ["는", "를", "이다", "으로", "에", "이라고", "라고", "와의", "인데"]

            # for josa in josas:
                # if text.endswith(josa):
                    # text = text[:-len(josa)]
                    # break

            # # temperature
            # if text.endswith("°"):
                # text += "C"

            # # etc
            # special_cases = ["(", ",", "였", "."]
            # for s in special_cases:
                # if text.endswith(s):
                    # text = text[:-len(s)]
            # return text

        def _clean_text(text):
            # https://github.com/allenai/document-qa/blob/2f9fa6878b60ed8a8a31bcf03f802cde292fe48b/docqa/data_processing/text_utils.py#L124
            # be consistent with quotes, and replace \u2014 and \u2212 which I have seen being mapped to UNK
            # by glove word vecs
            return (
                text.replace("''", '"')
                .replace("``", '"')
                .replace("\u2212", "-")
                .replace("\u2014", "\u2013")
            )

        predictions = {}
        topk_predictions = {}

        print("Read input_data...")
        data = self.data_handler.read(file_path)
        squad = json.loads(data)
        if "data" in squad:
            squad = squad["data"]

        wrong_count = 0

        print("Start predict 1-examples...")
        for article in tqdm(squad):
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]

                for qa in paragraph["qas"]:
                    question = qa["question"]
                    id_ = qa["id"]

                    # Marginal probabilities...
                    # prediction = self.get_predict_with_marginal(context, question)
                    prediction = self.get_predict(context, question)
                    # print("prediction count:", len(prediction))

                    topk_predictions[id_] = prediction
                    predictions[id_] = prediction[0]["text"]

                    # answer_texts = [q["text"] for q in qa["answers"]]

                    # # 1. Highest value
                    # sorted_prediction = sorted(prediction, key=lambda x: x["score"], reverse=True)
                    # prediction_text = sorted_prediction[0]["text"]

                    # 2. Cluster by text
                    # prediction_text = get_answer_after_clustering_marginal(prediction)
                    # prediction_text = post_processing(prediction_text)

                    # predictions[id_] = prediction_text
                    # if prediction_text not in answer_texts:
                        # pred_f1_score = metric_max_over_ground_truths(f1_score, prediction_text, answer_texts)

                        # if pred_f1_score <= 0.5:
                            # sorted_prediction = sorted(prediction, key=lambda x: x["score"], reverse=True)
                            # print("predict:", json.dumps(sorted_prediction[:5], indent=4, ensure_ascii=False))
                            # print("predict_text:", prediction_text)
                            # print("answers:", qa["answers"], "f1:", pred_f1_score)
                            # print("-"*50)
                        # wrong_count += 1

                    # is_answer = False
                    # for pred in prediction:
                        # if pred["text"] in answer_texts:
                            # predictions[id_] = pred["text"]
                            # is_answer = True
                            # break

                    # if not is_answer:
                        # prediction_text = sorted(prediction, key=lambda x: x["score"], reverse=True)[0]["text"]
                        # predictions[id_] = prediction_text

                        # print("predict:", prediction)
                        # print("predict_text:", prediction_text)
                        # print("answers:", qa["answers"])
                        # print("-"*50)
                        # wrong_count += 1

        print("total_count:", len(predictions), "wrong_count:", wrong_count)

        print("Completed...!")
        with open(output_path, "w") as out_file:
            out_file.write(json.dumps(topk_predictions, indent=4) + "\n")

        # Evaluate
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json
            if "data" in dataset:
                dataset = dataset["data"]
        # with open(output_path) as prediction_file:
            # predictions = json.load(prediction_file)

        results = evaluate(dataset, predictions)
        print(json.dumps(results))

    def get_predict(self, context, question):
        raw_feature = {"context": context, "question": question}
        # print(raw_feature)

        # Approach 1. Max Prob
        models = [
            (self.rc_experiment1, 0.94),
            # (self.rc_experiment2, 0.90)
            # (self.rc_experiment3, 0.85),
            # (self.rc_experiment4, 0.84),
        ]
        # models = [self.rc_experiment3, self.rc_experiment4]

        model = models[0][0]
        return sorted(model.predict(raw_feature), key=lambda x: x["score"], reverse=True)
