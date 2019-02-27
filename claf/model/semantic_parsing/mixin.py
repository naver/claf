
import torch

from claf.decorator import arguments_required
from claf.metric import wikisql_official
from claf.metric.wikisql_lib.dbengine import DBEngine
from claf.metric.wikisql_lib.query import Query


class WikiSQL:
    """
    WikiSQL Mixin Class
        with official evaluation

    * Args:
        token_embedder: 'TokenEmbedder'
    """

    AGG_OPS = ["None", "MAX", "MIN", "COUNT", "SUM", "AVG"]
    COND_OPS = ["EQL", "GT", "LT"]

    def make_metrics(self, predictions):
        """ aggregator, select_column, conditions accuracy """

        agg_accuracy, sel_accuracy, conds_accuracy = 0, 0, 0

        for index, pred in predictions.items():
            target = self._dataset.get_ground_truth(index)

            # Aggregator, Select_Column, Conditions
            agg_acc = 1 if pred["query"]["agg"] == target["agg_idx"] else 0
            sel_acc = 1 if pred["query"]["sel"] == target["sel_idx"] else 0

            pred_conds = pred["query"]["conds"]
            string_set_pred_conds = set(["#".join(map(str, cond)).lower() for cond in pred_conds])
            target_conds = [
                [target["conds_col"][i], target["conds_op"][i], target["conds_val_str"][i]]
                for i in range(target["conds_num"])
            ]
            string_set_target_conds = set(
                ["#".join(map(str, cond)).lower() for cond in target_conds]
            )

            conds_acc = (
                1 if string_set_pred_conds == string_set_target_conds else 0
            )  # not matter in order

            agg_accuracy += agg_acc
            sel_accuracy += sel_acc
            conds_accuracy += conds_acc

        total_count = len(self._dataset)

        agg_accuracy = 100.0 * agg_accuracy / total_count
        sel_accuracy = 100.0 * sel_accuracy / total_count
        conds_accuracy = 100.0 * conds_accuracy / total_count

        metrics = {
            "agg_accuracy": agg_accuracy,
            "sel_accuracy": sel_accuracy,
            "conds_accuracy": conds_accuracy,
        }

        self.write_predictions(predictions)

        wikisql_official_metrics = self._make_metrics_with_official(predictions)
        metrics.update(wikisql_official_metrics)
        return metrics

    def _make_metrics_with_official(self, preds):
        """
        WikiSQL official evaluation

        lf_accuracy: Logical-form accuracy
          - Directly compare the synthesized SQL query with the ground truth to
            check whether they match each other.
        ex_accuracy: Execution accuracy
          - Execute both the synthesized query and the ground truth query and
            compare whether the results match to each other.
        """

        labels = self._dataset.labels
        db_path = self._dataset.helper["db_path"]

        return wikisql_official.evaluate(labels, preds, db_path)

    def make_predictions(self, output_dict):
        predictions = {}
        sql_quries = self.generate_queries(output_dict)

        for i in range(len(sql_quries)):
            query = sql_quries[i]

            prediction = {}
            prediction.update(query)

            data_id = self._dataset.get_id(output_dict["data_id"][i])
            predictions[data_id] = prediction
        return predictions

    def generate_queries(self, output_dict):
        preds_agg = torch.argmax(output_dict["agg_logits"], dim=-1)
        preds_sel = torch.argmax(output_dict["sel_logits"], dim=-1)

        conds_logits = output_dict["conds_logits"]
        conds_num_logits, conds_column_logits, conds_op_logits, conds_value_logits = conds_logits

        preds_conds_num = torch.argmax(conds_num_logits, dim=-1)
        preds_conds_op = torch.argmax(conds_op_logits, dim=-1)

        sql_quries = []
        B = output_dict["agg_logits"].size(0)

        for i in range(B):
            if "table_id" in output_dict:
                table_id = output_dict["table_id"]
            else:
                table_id = self._dataset.get_table_id(output_dict["data_id"][i])

            query = {
                "table_id": table_id,
                "query": {"agg": preds_agg[i].item(), "sel": preds_sel[i].item()},
            }

            pred_conds_num = preds_conds_num[i].item()
            conds_pred = []
            if pred_conds_num == 0:
                pass
            else:
                _, pred_conds_column_idx = torch.topk(conds_column_logits[i], pred_conds_num)

                if preds_conds_op.dim() == 1:  # for one-example (TODO: fix hard-code)
                    pred_conds_op = preds_conds_op
                    conds_value_logits = conds_value_logits.squeeze(3)
                    conds_value_logits = conds_value_logits.squeeze(0)
                else:
                    pred_conds_op = preds_conds_op[i]

                if "tokenized_question" in output_dict:
                    tokenized_question = output_dict["tokenized_question"]
                else:
                    tokenized_question = self._dataset.get_tokenized_question(
                        output_dict["data_id"][i]
                    )

                conds_pred = [
                    [
                        pred_conds_column_idx[j].item(),
                        pred_conds_op[j].item(),
                        self.decode_pointer(tokenized_question, conds_value_logits[i][j]),
                    ]
                    for j in range(pred_conds_num)
                ]

            query["query"]["conds"] = conds_pred
            sql_quries.append(query)
        return sql_quries

    def decode_pointer(self, tokenized_question, cond_value_logits):
        question_text = " ".join(tokenized_question)
        tokenized_question = ["<BEG>"] + tokenized_question + ["<END>"]

        conds_value = []
        for value_logit in cond_value_logits:
            pred_value_pos = torch.argmax(value_logit[: len(tokenized_question)]).item()
            pred_value_token = tokenized_question[pred_value_pos]
            if pred_value_token == "<END>":
                break
            conds_value.append(pred_value_token)

        conds_value = self.merge_tokens(conds_value, question_text)
        return conds_value

    def merge_tokens(self, tok_list, raw_tok_str):
        lower_tok_str = raw_tok_str.lower()
        alphabet = set("abcdefghijklmnopqrstuvwxyz0123456789$(")
        special = {
            "-LRB-": "(",
            "-RRB-": ")",
            "-LSB-": "[",
            "-RSB-": "]",
            "``": '"',
            "''": '"',
            "--": "\u2013",
        }
        ret = ""
        double_quote_appear = 0
        for raw_tok in tok_list:
            if not raw_tok:
                continue
            tok = special.get(raw_tok, raw_tok)
            lower_tok = tok.lower()
            if tok == '"':
                double_quote_appear = 1 - double_quote_appear

            if len(ret) == 0:
                pass
            elif len(ret) > 0 and ret + " " + lower_tok in lower_tok_str:
                ret = ret + " "
            elif len(ret) > 0 and ret + lower_tok in lower_tok_str:
                pass
            elif lower_tok == '"':
                if double_quote_appear:
                    ret = ret + " "
            elif lower_tok[0] not in alphabet:
                pass
            elif (ret[-1] not in ["(", "/", "\u2013", "#", "$", "&"]) and (
                ret[-1] != '"' or not double_quote_appear
            ):
                ret = ret + " "
            ret = ret + tok
        return ret.strip()

    @arguments_required(["db_path", "table_id"])
    def predict(self, output_dict, arguments, helper):
        """
        Inference by raw_feature

        * Args:
            output_dict: model's output dictionary consisting of
            arguments: arguments dictionary consisting of user_input
            helper: dictionary for helping get answer

        * Returns:
            query: Generated SQL Query
            execute_result: Execute result by generated query
        """
        output_dict["table_id"] = arguments["table_id"]
        output_dict["tokenized_question"] = helper["tokenized_question"]

        prediction = self.generate_queries(output_dict)[0]
        pred_query = Query.from_dict(prediction["query"], ordered=True)

        dbengine = DBEngine(arguments["db_path"])
        try:
            pred_execute_result = dbengine.execute_query(
                prediction["table_id"], pred_query, lower=True
            )
        except IndexError as e:
            pred_execute_result = str(e)

        return {"query": str(pred_query), "execute_result": pred_execute_result}

    def print_examples(self, index, inputs, predictions):
        """
        Print evaluation examples

        * Args:
            index: data index
            inputs: mini-batch inputs
            predictions: prediction dictionary consisting of
                - key: 'id' (question id)
                - value: consisting of dictionary
                    table_id, query (agg, sel, conds)

        * Returns:
            print(Context, Question, Answers and Predict)
        """

        data_index = inputs["labels"]["data_idx"][index].item()
        data_id = self._dataset.get_id(data_index)

        helper = self._dataset.helper
        question = helper["examples"][data_id]["question"]

        label = self._dataset.get_ground_truth(data_id)

        dbengine = DBEngine(helper["db_path"])

        prediction = predictions[data_id]
        pred_query = Query.from_dict(prediction["query"], ordered=True)
        pred_execute_result = dbengine.execute_query(prediction["table_id"], pred_query, lower=True)

        print("- Question:", question)
        print("- Answers:")
        print("    SQL Query: ", label["sql_query"])
        print("    Execute Results:", label["execution_result"])
        print("- Predict:")
        print("    SQL Query: ", pred_query)
        print("    Execute Results:", pred_execute_result)
        print("-" * 30)
