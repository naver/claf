
import pytest

import spacy

from claf.tokens.tokenizer import BPETokenizer, CharTokenizer, SubwordTokenizer, WordTokenizer, SentTokenizer
from claf.tokens.tokenizer.utils import load_spacy_model_for_tokenizer


@pytest.fixture
def tokenizers(request):
    sent_name, sent_config, word_name, word_config, \
        subword_name, subword_config, char_name, char_config, \
        bpe_name, bpe_config = request.param

    sent_tokenizer = SentTokenizer(sent_name, config=sent_config)
    word_tokenizer = WordTokenizer(word_name, sent_tokenizer, config=word_config)
    subword_tokenizer = SubwordTokenizer(subword_name, word_tokenizer, config=subword_config)
    char_tokenizer = CharTokenizer(char_name, word_tokenizer, config=char_config)
    bpe_tokenizer = BPETokenizer(bpe_name, config=bpe_config)

    return {
        "sent": sent_tokenizer,
        "word": word_tokenizer,
        "subword": subword_tokenizer,
        "char": char_tokenizer,
        "bpe": bpe_tokenizer,
    }


@pytest.mark.parametrize("tokenizers", [(
    "punkt", {},
    "space_all", {},
    "wordpiece", {
        "vocab_path": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"
    },
    "character", {},
    "bpe", {})],
    indirect=True)
def test_en_character_tokenize(tokenizers):
    text = "Hello World"

    tokenizer = tokenizers["char"]
    results = tokenizer.tokenize(text)

    assert results == [["H", "e", "l", "l", "o"], ["W", "o", "r", "l", "d"]]


@pytest.mark.parametrize("tokenizers", [(
    "punkt", {},
    "space_all", {},
    "wordpiece", {
        "vocab_path": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"
    },
    "jamo_ko", {},
    "bpe", {})],
    indirect=True)
def test_jamo_ko_tokenize(tokenizers):
    text = "안녕 세상"

    tokenizer = tokenizers["char"]
    results = tokenizer.tokenize(text)
    assert results == [["ㅇ", "ㅏ", "ㄴ", "ㄴ", "ㅕ", "ㅇ"], ["ㅅ", "ㅔ", "ㅅ", "ㅏ", "ㅇ"]]


@pytest.mark.parametrize("tokenizers", [(
    "punkt", {},
    "bert_basic", {
        "do_lower_case": True
    },
    "wordpiece", {
        "vocab_path": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"
    },
    "jamo_ko", {},
    "bpe", {})],
    indirect=True)
def test_bert_uncased_en_tokenize(tokenizers):
    text = "expectancy of anyone"

    tokenizer = tokenizers["subword"]
    results = tokenizer.tokenize(text)
    assert results == ['expect', '##ancy', 'of', 'anyone']


@pytest.mark.parametrize("tokenizers", [(
    "punkt", {},
    "space_all", {},
    "wordpiece", {
        "vocab_path": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"
    },
    "character", {},
    "bpe", {})],
    indirect=True)
def test_space_all_tokenize(tokenizers):
    text = "Hi Hello\tHi\rHello\nHi"

    tokenizer = tokenizers["word"]
    results = tokenizer.tokenize(text)
    assert results == ['Hi', 'Hello', 'Hi', 'Hello', 'Hi']


@pytest.mark.parametrize("tokenizers", [(
    "punkt", {},
    "space_all", {},
    "wordpiece", {
        "vocab_path": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"
    },
    "character", {},
    "bpe", {})],
    indirect=True)
def test_punkt_tokenize(tokenizers):
    text = "Hello World. This is punkt tokenizer."

    tokenizer = tokenizers["sent"]
    results = tokenizer.tokenize(text)
    assert results == ['Hello World.', 'This is punkt tokenizer.']


@pytest.mark.parametrize("tokenizers", [(
    "punkt", {},
    "space_all", {},
    "wordpiece", {
        "vocab_path": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"
    },
    "character", {},
    "bpe", {})],
    indirect=True)
def test_word_with_regex_example_tokenize(tokenizers):
    text = "New York City:57–60 And Ted Ginn Jr.[citation needed]"

    sent_tokenizer = tokenizers["sent"]
    word_tokenizer = WordTokenizer("treebank_en", sent_tokenizer, split_with_regex=True)
    results = word_tokenizer.tokenize(text)
    print(results)
    assert results == ['New', 'York', 'City', ':', '57', '–', '60', 'And', 'Ted', 'Ginn', 'Jr', '.', '[', 'citation', 'needed', ']']


@pytest.mark.parametrize("tokenizers", [(
    "punkt", {},
    "space_all", {},
    "wordpiece", {
        "vocab_path": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"
    },
    "character", {},
    "bpe", {})],
    indirect=True)
def test_spacy_model_with_regex_example_tokenize(tokenizers):
    text = "In 1096, Crusaders passing by the siege of Amalfi were joined by Bohemond of Taranto and his nephew Tancred with an army of Italo-Normans. Bohemond was the de facto leader of the Crusade during its passage through Asia Minor. After the successful Siege of Antioch in 1097, Bohemond began carving out an independent principality around that city. Tancred was instrumental in the conquest of Jerusalem and he worked for the expansion of the Crusader kingdom in Transjordan and the region of Galilee.[citation needed]"

    sent_tokenizer = SentTokenizer("punkt")
    word_tokenizer = WordTokenizer("spacy_en", sent_tokenizer, split_with_regex=True)

    disables = ["vectors", "textcat", "parser"]
    spacy_model = spacy.load("en_core_web_sm", disable=disables)
    spacy_model.tokenizer = load_spacy_model_for_tokenizer(
        word_tokenizer.extra_split_chars_re
    )

    sentences = sent_tokenizer.tokenize(text)

    spacy_model_results = []
    for sentence in sentences:
        spacy_model_results += [token.text for token in spacy_model(sentence)]

    assert word_tokenizer.tokenize(text) == spacy_model_results

    text = "20th Century Fox, Lionsgate, Paramount Pictures, Universal Studios and Walt Disney Studios paid for movie trailers to be aired during the Super Bowl. Fox paid for Deadpool, X-Men: Apocalypse, Independence Day: Resurgence and Eddie the Eagle, Lionsgate paid for Gods of Egypt, Paramount paid for Teenage Mutant Ninja Turtles: Out of the Shadows and 10 Cloverfield Lane, Universal paid for The Secret Life of Pets and the debut trailer for Jason Bourne and Disney paid for Captain America: Civil War, The Jungle Book and Alice Through the Looking Glass.[citation needed]"
    sentences = sent_tokenizer.tokenize(text)

    spacy_model_results = []
    for sentence in sentences:
        spacy_model_results += [token.text for token in spacy_model(sentence)]

    assert word_tokenizer.tokenize(text) == spacy_model_results


@pytest.mark.parametrize("tokenizers", [(
    "punkt", {},
    "space_all", {},
    "wordpiece", {
        "vocab_path": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"
    },
    "character", {},
    "roberta", {
        "vocab_path": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json",
        "merges_path": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt"
    })],
    indirect=True)
def test_bpe_tokenize(tokenizers):
    text = "As you eat the most, you want the least."

    tokenizer = tokenizers["bpe"]
    results = tokenizer.tokenize(text)
    assert results == ['As', 'Ġyou', 'Ġeat', 'Ġthe', 'Ġmost', ',', 'Ġyou', 'Ġwant', 'Ġthe', 'Ġleast', '.']
