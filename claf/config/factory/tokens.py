
from overrides import overrides

from claf.config.registry import Registry
from claf.config.utils import convert_config2dict
from claf.tokens import tokenizer

from .base import Factory


def make_tokenizer(tokenizer_cls, tokenizer_config, parent_tokenizers={}):
    if tokenizer_config is None or "name" not in tokenizer_config:
        return None

    package_name = tokenizer_config["name"]
    package_config = tokenizer_config.get(package_name, {})
    tokenizer_config["config"] = package_config
    if package_name in tokenizer_config:
        del tokenizer_config[package_name]

    tokenizer_config.update(parent_tokenizers)

    return tokenizer_cls(**tokenizer_config)


def make_all_tokenizers(all_tokenizer_config):
    """ Tokenizer is resource used all token together """

    sent_tokenizer = make_tokenizer(
        tokenizer.SentTokenizer, all_tokenizer_config.get("sent", {"name": "punkt"})
    )
    word_tokenizer = make_tokenizer(
        tokenizer.WordTokenizer,
        all_tokenizer_config.get("word", None),
        parent_tokenizers={"sent_tokenizer": sent_tokenizer},
    )
    subword_tokenizer = make_tokenizer(
        tokenizer.SubwordTokenizer,
        all_tokenizer_config.get("subword", None),
        parent_tokenizers={"word_tokenizer": word_tokenizer},
    )
    char_tokenizer = make_tokenizer(
        tokenizer.CharTokenizer,
        all_tokenizer_config.get("char", None),
        parent_tokenizers={"word_tokenizer": word_tokenizer},
    )

    return {
        "char": char_tokenizer,
        "subword": subword_tokenizer,
        "word": word_tokenizer,
        "sent": sent_tokenizer,
    }


class TokenMakersFactory(Factory):
    """
    TokenMakers Factory Class

    * Args:
        config: token config from argument (config.token)
    """

    LANGS = ["eng", "kor"]

    def __init__(self, config):
        self.config = config
        self.registry = Registry()

    @overrides
    def create(self):
        tokenizers = make_all_tokenizers(convert_config2dict(self.config.tokenizer))
        token_names, token_types = self.config.names, self.config.types

        if len(token_names) != len(token_types):
            raise ValueError("token_names and token_types must be same length.")

        token_makers = {"tokenizers": tokenizers}
        for token_name, token_type in sorted(zip(token_names, token_types)):
            token_config = getattr(self.config, token_name, {})
            if token_config != {}:
                token_config = convert_config2dict(token_config)

            # Token (tokenizer, indexer, embedding, vocab)
            token_config = {
                "tokenizers": tokenizers,
                "indexer_config": token_config.get("indexer", {}),
                "embedding_config": token_config.get("embedding", {}),
                "vocab_config": token_config.get("vocab", {}),
            }
            token_makers[token_name] = self.registry.get(f"token:{token_type}")(**token_config)
        return token_makers
