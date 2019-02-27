
import json
import logging
import os

from tqdm import tqdm


logger = logging.getLogger(__name__)


def read_wiki_articles(dir_path):
    """
    WikiExtractor's output like below:
    (https://github.com/attardi/wikiextractor)

    wiki_path/
      - AA
        - wiki_00
        - wiki_01
        ...
      - AB
        ...
    """
    dir_paths = get_subdir_paths(dir_path)

    all_file_path = []
    for path in dir_paths:
        all_file_path += get_file_paths(path)

    articles = []
    for path in tqdm(all_file_path, desc="Read Wiki Articles"):
        articles += read_wiki_article(path)
    return articles


def get_subdir_paths(dir_path):
    dir_paths = []

    for path, subdirs, __ in os.walk(dir_path):
        for dir_name in subdirs:
            dir_paths.append(os.path.join(path, dir_name))
    return dir_paths


def get_file_paths(dir_path):
    file_paths = []

    for path, _, files in os.walk(dir_path):
        for file_name in files:
            file_paths.append(os.path.join(path, file_name))
    return file_paths


def read_wiki_article(file_path):
    """
    Wiki articles format (WikiExtractor)
    => {"id": "", "revid": "", "url":"", "title": "", "text": "..."}
    """

    articles = []
    with open(file_path, "r", encoding="utf-8") as in_file:
        for line in in_file.readlines():
            article = json.loads(line)
            articles.append(article)

    return [WikiArticle(**article) for article in articles]


class WikiArticle:  # pragma: no cover
    def __init__(self, id=None, url=None, title=None, text=None):
        self._id = id
        self._url = url
        self._title = title
        self._text = text

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id):
        self._id = id

    @property
    def url(self):
        return self._url

    @url.setter
    def url(self, url):
        self._url = url

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, title):
        self._title = title

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, text):
        self._text = text
