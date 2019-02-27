
import spacy


def create_tokenizer_with_regex(nlp, split_regex):
    prefixes_re = spacy.util.compile_prefix_regex(nlp.Defaults.prefixes)
    infix_re = split_regex
    suffix_re = spacy.util.compile_suffix_regex(nlp.Defaults.suffixes)

    return spacy.tokenizer.Tokenizer(
        nlp.vocab,
        nlp.Defaults.tokenizer_exceptions,
        prefix_search=prefixes_re.search,
        infix_finditer=infix_re.finditer,
        suffix_search=suffix_re.search,
        token_match=None,
    )


def load_spacy_model_for_tokenizer(split_regex):
    model = spacy.load("en_core_web_sm", disable=["vectors", "textcat", "tagger", "parser", "ner"])

    if split_regex is not None:
        spacy_tokenizer = create_tokenizer_with_regex(model, split_regex)
        model.tokenizer = spacy_tokenizer
    return model
