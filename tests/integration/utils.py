
import json
import os
import random
import shutil


def make_squad_synthetic_data(output_path):
    ANSWER_TOKEN = "ANSWER"
    DATA_SIZE = 10

    out_squad = {'data': [], 'version': "0.1"}
    article = {
        "paragraphs": [],
        "title": "Synthetic data for test"
    }

    for _ in range(DATA_SIZE):
        token_count = random.randint(10, 20)
        qas = []
        query_count = 10
        answers = [{"answer_start": 0, "answer_end": 0, "text": ANSWER_TOKEN}]
        qa = {
            "id": f"{token_count}_{query_count}",
            "answers": answers,
            "question": make_random_tokens(query_count)
        }
        qas.append(qa)
        paragraph = {
            "context": make_random_tokens(token_count, answer_token=ANSWER_TOKEN),
            "qas": qas
        }
        article["paragraphs"].append(paragraph)
    out_squad['data'].append(article)

    dir_path = os.path.dirname(output_path)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path, ignore_errors=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as fp:
        json.dump(out_squad, fp)


def make_wiki_article_synthetic_data(output_dir):
    AA_articles = [
        {"id": 0, "url": "url", "title": "title1", "text": make_random_tokens(10)},
        {"id": 1, "url": "url", "title": "title2", "text": make_random_tokens(10)},
        {"id": 2, "url": "url", "title": "title3", "text": make_random_tokens(10)},
    ]
    AA_articles = [json.dumps(item) for item in AA_articles]

    AA_path = os.path.join(output_dir, "AA", "wiki_00")
    print(AA_path)
    os.makedirs(os.path.dirname(AA_path), exist_ok=True)
    with open(AA_path, "w", encoding="utf-8") as out_file:
        out_file.write("\n".join(AA_articles))

    assert os.path.exists(AA_path) == True

    AB_articles = [
        {"id": 3, "url": "url", "title": "title4", "text": make_random_tokens(10)},
        {"id": 4, "url": "url", "title": "title5", "text": make_random_tokens(10)},
        {"id": 5, "url": "url", "title": "title6", "text": make_random_tokens(10)},
    ]
    AB_articles = [json.dumps(item) for item in AB_articles]

    AB_path = os.path.join(output_dir, "AB", "wiki_00")
    os.makedirs(os.path.dirname(AB_path), exist_ok=True)
    with open(AB_path, "w", encoding="utf-8") as out_file:
        out_file.write("\n".join(AB_articles))

    assert os.path.exists(AB_path) == True


def make_random_tokens(length, answer_token=""):
    tokens = ['kox', 'pev', 'hi', 'shemini', 'outvote']

    if answer_token:
        output = [answer_token]
    else:
        output = []
    for _ in range(length-1):
        output.append(random.choice(tokens))
    return " ".join(output)


def make_seq_cls_synthetic_data(output_path):
    class_key = "label"
    classes = ["foo", "bar", "baz", "qux", "quux", "corge", "grault", "graply", "waldo"]
    data_size = 10

    out_seq_cls = {
        "data": [],
        class_key: classes,
    }

    for _ in range(data_size):
        token_count = random.randint(10, 20)
        sequence = make_random_tokens(token_count)
        class_ = random.choice(classes)

        out_seq_cls["data"].append({
            "sequence": sequence,
            class_key: class_,
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as fp:
        json.dump(out_seq_cls, fp)


def make_tok_cls_synthetic_data(output_path):
    tag_key = "label"
    tags = ["O", "foo", "bar", "baz", "qux"]
    data_size = 10

    out_tok_cls = {
        "data": [],
        tag_key: ["O"] + [f"{prefix}-{tag}" for prefix in ["B", "I"] for tag in tags]
    }

    for _ in range(data_size):
        token_count = random.randint(10, 20)
        sequence = make_random_tokens(token_count)
        tag_sequence = make_dummy_tags(sequence, tags)

        out_tok_cls["data"].append({
            "sequence": sequence,
            tag_key: tag_sequence,
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as fp:
        json.dump(out_tok_cls, fp)


def make_dummy_tags(sequence, dummy_tag_cands):
    words = sequence.split()

    tags = []
    prev_tag = None
    for word in words:
        if random.random() < 0.3:
            tag = "O"
        else:
            tag = random.choice(dummy_tag_cands)
            if prev_tag is None or prev_tag[2:] != tag:
                tag = "B-" + tag
            else:
                tag = "I-" + tag
        tags.append(tag)
        if tag == "O":
            prev_tag = None
        else:
            prev_tag = tag

    return tags
