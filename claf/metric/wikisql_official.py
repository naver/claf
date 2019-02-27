""" Official evaluation script for WikiSQL dataset. """

import json
from argparse import ArgumentParser
from tqdm import tqdm
from claf.metric.wikisql_lib.dbengine import DBEngine
from claf.metric.wikisql_lib.query import Query


def count_lines(fname):  # pragma: no cover
    with open(fname) as f:
        return sum(1 for line in f)


def evaluate(labels, predictions, db_path, ordered=True):  # pragma: no cover
    """ labels and predictions: dictionary {data_uid: sql_data, ...} """
    engine = DBEngine(db_path)

    exact_match, grades = [], []
    for idx, data_uid in enumerate(predictions):
        eg = labels[data_uid]
        ep = predictions[data_uid]

        qg = eg["sql_query"]
        gold = eg["execution_result"]

        pred = ep.get("error", None)
        qp = None
        if not ep.get("error", None):
            try:
                qp = Query.from_dict(ep["query"], ordered=ordered)
                pred = engine.execute_query(ep["table_id"], qp, lower=True)
            except Exception as e:
                pred = repr(e)

        correct = pred == gold
        match = qp == qg
        grades.append(correct)
        exact_match.append(match)

    return {
        "ex_accuracy": sum(grades) / len(grades) * 100.0,
        "lf_accuracy": sum(exact_match) / len(exact_match) * 100.0,
    }


if __name__ == "__main__":  # pragma: no cover
    parser = ArgumentParser()
    parser.add_argument("source_file", help="source file for the prediction")
    parser.add_argument("db_file", help="source database for the prediction")
    parser.add_argument("pred_file", help="predictions by the model")
    parser.add_argument(
        "--ordered",
        action="store_true",
        help="whether the exact match should consider the order of conditions",
    )
    args = parser.parse_args()

    engine = DBEngine(args.db_file)
    exact_match = []
    with open(args.source_file) as fs, open(args.pred_file) as fp:
        grades = []
        for ls, lp in tqdm(zip(fs, fp), total=count_lines(args.source_file)):
            eg = json.loads(ls)
            ep = json.loads(lp)
            qg = Query.from_dict(eg["sql"], ordered=args.ordered)
            gold = engine.execute_query(eg["table_id"], qg, lower=True)
            pred = ep.get("error", None)
            qp = None
            if not ep.get("error", None):
                try:
                    qp = Query.from_dict(ep["query"], ordered=args.ordered)
                    pred = engine.execute_query(eg["table_id"], qp, lower=True)
                except Exception as e:
                    pred = repr(e)
            correct = pred == gold
            match = qp == qg
            grades.append(correct)
            exact_match.append(match)
        print(
            json.dumps(
                {
                    "ex_accuracy": sum(grades) / len(grades),
                    "lf_accuracy": sum(exact_match) / len(exact_match),
                },
                indent=2,
            )
        )
