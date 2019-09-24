
import logging

from claf.model.sequence_classification.mixin import SequenceClassification

logger = logging.getLogger(__name__)


class MultiTask:
    """ MultiTask Mixin Class """

    CLASSIFICATION = "classification"

    def make_predictions(self, output_dict):
        mixin_obj = None
        if self.curr_task_category == self.CLASSIFICATION:
            mixin_obj = SequenceClassification()

        self._set_model_properties(mixin_obj)
        predictions = mixin_obj.make_predictions(output_dict)
        for k, v in predictions.items():
            predictions[k]["task_index"] = output_dict["task_index"]
        return predictions

    def predict(self, output_dict, arguments, helper):
        if self.curr_task_category == self.CLASSIFICATION:
            mixin_obj = SequenceClassification()

        self._set_model_properties(mixin_obj)
        return mixin_obj.predict(output_dict, arguments, helper)

    def make_metrics(self, predictions):
        # split predictions by task_index -> each task make_metrics then add task_index as prefix
        task_predictions = [{} for _ in range(len(self.tasks))]  # init
        for k, v in predictions.items():
            task_index = v["task_index"]
            task_predictions[task_index][k] = v

        all_metrics = {}
        for task_index, predictions in enumerate(task_predictions):
            if self.curr_task_category == self.CLASSIFICATION:
                mixin_obj = SequenceClassification()
            self._set_model_properties(mixin_obj)

            task_metrics = mixin_obj.make_metrics(predictions)
            for k, v in task_metrics.items():
                all_metrics[f"task-{task_index}/{k}"] = v
        return all_metrics

    def write_predictions(self, predictions, file_path=None, is_dict=True):
        if self.curr_task_category == self.CLASSIFICATION:
            mixin_obj = SequenceClassification()

        # TODO: split predictions by task_index -> each task make_metrics then add task_index as prefix
        self._set_model_properties(mixin_obj)
        return mixin_obj.write_predictions(predictions, file_path=file_path, is_dict=is_dict)

    def _set_model_properties(self, mixin_obj):
        mixin_obj._config = self.config
        mixin_obj._log_dir = self.log_dir
        mixin_obj._dataset = self.curr_dataset
        mixin_obj._train_counter = self.train_counter
        mixin_obj.training = self.training
        mixin_obj._vocabs = self.vocabs
