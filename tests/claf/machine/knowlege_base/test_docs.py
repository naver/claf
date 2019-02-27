
import json
import os

from claf.machine.knowlege_base.docs import read_wiki_articles


def test_read_wiki_articles():
    articles = [
        {"id": 0 , "url": "url", "title": "title", "text": "text"},
        {"id": 1 , "url": "url", "title": "title", "text": "text"},
        {"id": 2 , "url": "url", "title": "title", "text": "text"},
    ]

    file_path = "./wiki_articles.json"
    with open(file_path, "w", encoding="utf-8") as out_file:
        for article in articles:
            out_file.write(json.dumps(article))

    articles = read_wiki_articles(file_path)
    os.remove(file_path)
