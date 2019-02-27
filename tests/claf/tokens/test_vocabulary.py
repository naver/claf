
from collections import Counter
import os

from claf.tokens.vocabulary import Vocab


def test_init_vocab():
    vocab = Vocab("token_name")
    vocab.init()

    assert vocab.get_all_tokens() == ["[PAD]", "[UNK]"]


def test_init_vocab_with_special_token():
    vocab = Vocab("token_name", start_token="<s>", end_token="</s>", cls_token="[CLS]", sep_token="[SEP]")
    vocab.init()

    assert vocab.get_all_tokens() == ["[PAD]", "[UNK]", "<s>", "</s>", "[CLS]", "[SEP]"]


def test_from_texts():
    texts = "A\nB\nC\nD"

    vocab = Vocab("token_name")
    vocab.from_texts(texts)

    assert vocab.get_all_tokens() == ["A", "B", "C", "D", "[PAD]", "[UNK]"]


def test_from_texts_with_pad():
    texts = "<pad>\nA\nB\nC\nD"

    vocab = Vocab("token_name", pad_token="<pad>")
    vocab.from_texts(texts)

    assert vocab.get_all_tokens() == ["<pad>", "A", "B", "C", "D", "[UNK]"]


def test_from_texts_with_pad_but_not_define():
    texts = "<pad>\nA\nB\nC\nD"

    vocab = Vocab("token_name")
    vocab.from_texts(texts)

    assert vocab.get_all_tokens() == ["<pad>", "A", "B", "C", "D", "[PAD]", "[UNK]"]


def test_build():
    tokens = ["A", "A", "A", "B", "B"]
    token_counter = Counter(tokens)

    vocab = Vocab("token_name")
    vocab.build(token_counter)

    assert vocab.get_all_tokens() == ["[PAD]", "[UNK]", "A", "B"]


def test_build_with_max_vocab_size():
    tokens = ["A", "A", "A", "B", "B"]
    token_counter = Counter(tokens)

    vocab = Vocab("token_name", max_vocab_size=1)
    vocab.build(token_counter)

    assert vocab.get_all_tokens() == ["[PAD]", "[UNK]", "A"]


def test_build_with_min_count():
    tokens = ["A", "A", "A", "B", "B"]
    token_counter = Counter(tokens)

    vocab = Vocab("token_name", min_count=3)
    vocab.build(token_counter)

    assert vocab.get_all_tokens() == ["[PAD]", "[UNK]", "A"]


def test_get_token():
    texts = "A\nB\nC\nD"

    vocab = Vocab("token_name")
    vocab.from_texts(texts)

    assert vocab.get_all_tokens() == ["A", "B", "C", "D", "[PAD]", "[UNK]"]
    assert vocab.get_token(2) == "C"


def test_save_and_load():
    texts = "A\nB\nC\nD"

    vocab = Vocab("token_name")
    vocab.from_texts(texts)

    vocab_path = "./test_vocab.txt"
    vocab.dump(vocab_path)

    vocab2 = Vocab("token_name")
    vocab2.load(vocab_path)

    os.remove(vocab_path)
    assert vocab.get_all_tokens() == vocab2.get_all_tokens()


def test_build_with_pretrained_file_all():
    texts = "[PAD]\n[UNK]\nA\nB\nC\nD"

    vocab_path = "./test_vocab.txt"
    with open(vocab_path, "w", encoding="utf-8") as out_file:
        out_file.write(texts)

    vocab = Vocab("token_name", pretrained_path=vocab_path, pretrained_token=Vocab.PRETRAINED_ALL)

    token_counter = None
    vocab.build_with_pretrained_file(token_counter)

    os.remove(vocab_path)
    assert vocab.get_all_tokens() == ["[PAD]", "[UNK]", "A", "B", "C", "D"]


def test_build_with_pretrained_file_intersect():
    texts = "[PAD]\n[UNK]\nA\nB\nC\nD"

    vocab_path = "./test_vocab.txt"
    with open(vocab_path, "w", encoding="utf-8") as out_file:
        out_file.write(texts)

    vocab = Vocab("token_name", pretrained_path=vocab_path, pretrained_token=Vocab.PRETRAINED_INTERSECT)

    input_texts = ["B", "C", "D", "E"]
    token_counter = Counter(input_texts)
    vocab.build_with_pretrained_file(token_counter)

    os.remove(vocab_path)
    assert vocab.get_all_tokens() == ["[PAD]", "[UNK]", "B", "C", "D"]
