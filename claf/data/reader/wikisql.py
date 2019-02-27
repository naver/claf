
import json
import logging
from pathlib import Path
import uuid

from overrides import overrides
from tqdm import tqdm

from claf.data.dataset import WikiSQLDataset
from claf.data.batch import make_batch
from claf.data.reader.base import DataReader
from claf.decorator import register
from claf.metric.wikisql_lib.dbengine import DBEngine
from claf.metric.wikisql_lib.query import Query

logger = logging.getLogger(__name__)


@register("reader:wikisql")
class WikiSQLReader(DataReader):
    """
    WikiSQL DataReader
    (http://arxiv.org/abs/1709.00103)

    * Args:
        file_paths: .json file paths (train and dev)
        tokenizers: defined tokenizers config (char/word)
    """

    def __init__(self, file_paths, tokenizers, context_max_length=None, is_test=None):
        super(WikiSQLReader, self).__init__(file_paths, WikiSQLDataset)
        self.is_test = is_test
        self.text_columns = ["column", "question"]

        if "word" not in tokenizers:
            raise ValueError("WordTokenizer is required. define English WordTokenizer")
        self.word_tokenizer = tokenizers["word"]
        self.dbengine = None

    @overrides
    def _read(self, file_path, data_type=None):
        file_path = self.data_handler.read(file_path, return_path=True)
        file_path = Path(file_path)

        data_dir = file_path.parent
        file_name = file_path.stem

        db_path = data_dir / f"{file_name}.db"
        table_path = data_dir / f"{file_name}.tables.jsonl"

        self.dbengine = DBEngine(db_path)

        helper = {"file_path": file_path, "db_path": db_path, "examples": {}}
        features, labels = [], []

        sql_datas, table_data = self.load_data(file_path, table_path, data_type=data_type)
        for sql_data in tqdm(sql_datas, desc=data_type):
            question = sql_data["question"]
            table_id = sql_data["table_id"]
            column_headers = table_data[table_id]["header"]

            feature_row = {"column": column_headers, "question": question}

            data_uid = str(uuid.uuid1())
            conditions_value_position = self.get_coditions_value_position(
                sql_data["question"], [x[2] for x in sql_data["sql"]["conds"]]
            )

            sql_query = Query.from_dict(sql_data["sql"], ordered=True)
            execution_result = self.dbengine.execute_query(table_id, sql_query, lower=True)

            label_row = {
                "id": data_uid,
                "table_id": table_id,
                "tokenized_question": self.word_tokenizer.tokenize(question),
                "aggregator_idx": sql_data["sql"]["agg"],
                "select_column_idx": sql_data["sql"]["sel"],
                "conditions_num": len(sql_data["sql"]["conds"]),
                "conditions_column_idx": [x[0] for x in sql_data["sql"]["conds"]],
                "conditions_operator_idx": [x[1] for x in sql_data["sql"]["conds"]],
                "conditions_value_string": [str(x[2]) for x in sql_data["sql"]["conds"]],
                "conditions_value_position": conditions_value_position,
                "sql_query": sql_query,
                "execution_result": execution_result,
            }

            features.append(feature_row)
            labels.append(label_row)

            helper["examples"][data_uid] = {
                "question": question,
                "sql_query": sql_query,
                "execution_result": execution_result,
            }

            if self.is_test and len(labels) == 10:
                break

        return make_batch(features, labels), helper

    @overrides
    def read_one_example(self, inputs):
        """ inputs keys: question, column, db_path, table_id """
        question_text = inputs["question"]
        helper = {"tokenized_question": self.word_tokenizer.tokenize(question_text)}
        return inputs, helper

    def load_data(self, sql_path, table_path, data_type=None):
        sql_data = []
        table_data = {}

        logger.info(f"Loading data from {sql_path}")
        with open(sql_path) as inf:
            for line in tqdm(inf, desc=f"sql_{data_type}"):
                sql = json.loads(line.strip())
                sql_data.append(sql)

        logger.info(f"Loading data from {table_path}")
        with open(table_path) as inf:
            for line in tqdm(inf, desc=f"table_{data_type}"):
                tab = json.loads(line.strip())
                table_data[tab["id"]] = tab

        for sql in sql_data:
            assert sql["table_id"] in table_data
        return sql_data, table_data

    def get_coditions_value_position(self, question, values):
        tokenized_question = self.word_tokenizer.tokenize(question.lower())
        tokenized_values = [self.word_tokenizer.tokenize(str(value).lower()) for value in values]

        START_TOKEN, END_TOKEN = "<BEG>", "<END>"

        token_to_index = {START_TOKEN: 0}
        for token in tokenized_question:
            token_to_index[token] = len(token_to_index)
        token_to_index[END_TOKEN] = len(token_to_index)

        position_tokens = []
        for value in tokenized_values:
            position_token = [token_to_index[START_TOKEN]]
            for token in value:
                if token in token_to_index:
                    position_token.append(token_to_index[token])
                else:
                    for i in range(len(tokenized_question)):
                        q_token = tokenized_question[i]
                        if token in q_token:
                            position_token.append(token_to_index[q_token])
            position_token.append(token_to_index[END_TOKEN])

            assert len(position_token) != 2
            position_tokens.append(position_token)

        return position_tokens
