
import logging

from claf.model.sequence_classification.mixin import SequenceClassification
from claf.model.regression.mixin import Regression

logger = logging.getLogger(__name__)


class MultiTask:
    """ MultiTask Mixin Class """

    CLASSIFICATION = "classification"
    REGRESSION = "regression"

    def make_predictions(self, output_dict):
        mixin_obj = None
        task_index = output_dict["task_index"].item()
        task_category = self.tasks[task_index]["category"]
        if task_category == self.CLASSIFICATION:
            mixin_obj = SequenceClassification()
        elif task_category == self.REGRESSION:
            mixin_obj = Regression()
        else:
            raise ValueError("task category error.")

        self._set_model_properties(mixin_obj, task_index=task_index)
        predictions = mixin_obj.make_predictions(output_dict)
        for k, v in predictions.items():
            predictions[k]["task_index"] = output_dict["task_index"]
        return predictions

    def predict(self, output_dict, arguments, helper):
        task_index = output_dict["task_index"].item()
        task_category = self.tasks[task_index]["category"]
        if task_category == self.CLASSIFICATION:
            mixin_obj = SequenceClassification()
        elif task_category == self.REGRESSION:
            mixin_obj = Regression()
        else:
            raise ValueError("task category error.")

        self._set_model_properties(mixin_obj, task_index=task_index)
        return mixin_obj.predict(output_dict, arguments, helper)

    def make_metrics(self, predictions):
        # split predictions by task_index -> each task make_metrics then add task_index as prefix
        task_predictions = [{} for _ in range(len(self.tasks))]  # init
        for k, v in predictions.items():
            task_index = v["task_index"]
            task_predictions[task_index][k] = v

        all_metrics = {"average": 0}
        for task_index, predictions in enumerate(task_predictions):
            task_category = self.tasks[task_index]["category"]
            if task_category == self.CLASSIFICATION:
                mixin_obj = SequenceClassification()
            elif task_category == self.REGRESSION:
                mixin_obj = Regression()
            else:
                raise ValueError("task category error.")

            self._set_model_properties(mixin_obj, task_index=task_index)

            task_metrics = mixin_obj.make_metrics(predictions)
            for k, v in task_metrics.items():
                task_name = self.tasks[task_index]["name"].replace("_bert", "")  # hard_code
                all_metrics[f"{task_name}/{k}"] = v

                task_metric_key = self.tasks[task_index]["metric_key"]
                if k == task_metric_key:
                    all_metrics["average"] += v

        all_metrics["average"] /= len(task_predictions)
        return all_metrics

    def write_predictions(self, predictions, file_path=None, is_dict=True):
        pass
        # if self.curr_task_category == self.CLASSIFICATION:
            # mixin_obj = SequenceClassification()
        # elif self.curr_task_category == self.REGRESSION:
            # mixin_obj = Regression()
        # else:
            # raise ValueError("task category error.")

        # # TODO: split predictions by task_index -> each task make_metrics then add task_index as prefix
        # self._set_model_properties(mixin_obj)
        # return mixin_obj.write_predictions(predictions, file_path=file_path, is_dict=is_dict)

    def _set_model_properties(self, mixin_obj, task_index=None):
        mixin_obj._config = self.config
        mixin_obj._log_dir = self.log_dir
        if task_index is None:
            mixin_obj._dataset = self.curr_dataset
        else:
            mixin_obj._dataset = self._dataset.task_datasets[task_index]
        mixin_obj._train_counter = self.train_counter
        mixin_obj.training = self.training
        mixin_obj._vocabs = self.vocabs
