
import argparse
import json
import os
import random
import uuid


def make_squad_synthetic_data(output_path, max_context_length, question_lengths):
    ANSWER_TOKEN = "ANSWER"

    out_squad = {'data': [], 'version': "0.1"}
    article = {
        "paragraphs": [],
        "title": "Synthetic data for test"
    }

    for token_count in range(10, max_context_length):
        qas = []
        for question_length in question_lengths:
            answers = [{"answer_start": 0, "answer_end": 0, "text": ANSWER_TOKEN}]
            qa = {
                "id": str(uuid.uuid1()),
                "answers": answers,
                "question": make_random_tokens(question_length)
            }
            qas.append(qa)

        paragraph = {
            "context": make_random_tokens(token_count, answer_token=ANSWER_TOKEN),
            "qas": qas
        }
        article["paragraphs"].append(paragraph)
    out_squad['data'].append(article)

    with open(output_path, 'w') as fp:
        json.dump(out_squad, fp)


def make_random_tokens(length, answer_token=""):
    tokens = ['kox', 'pev', 'hi', 'shemini', 'outvote']

    if answer_token:
        output = [answer_token]
    else:
        output = []
    for _ in range(length-1):
        output.append(random.choice(tokens))
    return " ".join(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=str,
                        help='synthetic data output path')
    parser.add_argument('--max_context_length', type=int,
                        help='The number of maximum context length')
    parser.add_argument('--question_lengths', nargs="+", type=int,
                        help='The numbers of question length')
    args = parser.parse_args()

    make_squad_synthetic_data(args.output_path, args.max_context_length, args.question_lengths)
