
from overrides import overrides
import spacy

from claf.tokens.linguistic import POSTag, NER

from .base import TokenIndexer


class LinguisticIndexer(TokenIndexer):
    """
    Linguistic Token Indexer

    * Property
        vocab: Vocab (claf.tokens.vocabulary)

    * Args:
        tokenizer: WordTokenizer

    * Kwargs:
        pos_tag: POS Tagging
        ner: Named Entity Recognition
        dep: Dependency Parser
    """

    def __init__(self, tokenizer, pos_tag=None, ner=None, dep=None):
        super(LinguisticIndexer, self).__init__(tokenizer)

        self.spacy_model = None

        # Features
        self.use_pos_tag = pos_tag
        self.pos_to_index = {t: i for i, t in enumerate(POSTag.classes)}

        self.use_ner = ner
        self.ner_to_index = {t: i for i, t in enumerate(NER.classes)}

        self.use_dep = dep
        if dep:
            raise NotImplementedError("Dependency Parser feature")

    @overrides
    def index(self, text):
        package = self.tokenizer.name
        return getattr(self, f"_{package}")(text)

    """ Need to match with Tokenizer's package """

    def _mecab_ko(self, text):
        raise NotImplementedError("Linguistic Feature with mecab package")

    def _nltk_en(self, text):
        raise NotImplementedError("Linguistic Feature with nltk package")

    def _spacy_en(self, text):
        if self.spacy_model is None:
            from claf.tokens.tokenizer.utils import load_spacy_model_for_tokenizer

            disables = ["vectors", "textcat", "parser"]
            if not self.use_pos_tag:
                disables.apppend("tagger")
            if not self.use_ner:
                disables.apppend("ner")

            self.spacy_model = spacy.load("en_core_web_sm", disable=disables)
            self.spacy_model.tokenizer = load_spacy_model_for_tokenizer(
                self.tokenizer.extra_split_chars_re
            )

        sent_tokenizer = self.tokenizer.sent_tokenizer
        sentences = sent_tokenizer.tokenize(text)

        ner_entities = {}
        docs = []
        for sentence in sentences:
            doc = self.spacy_model(sentence)
            docs.append(doc)

            if self.use_ner:
                for e in doc.ents:
                    ner_entities[e.text] = e.label_

        linguistic_features = []
        for doc in docs:
            for token in doc:
                if token.is_space:
                    continue

                feature = []
                if self.use_pos_tag:
                    feature.append(self.pos_to_index[token.pos_])
                if self.use_ner:
                    feature.append(self.ner_to_index[ner_entities.get(token.text, "NONE")])

                linguistic_features.append(feature)
        return linguistic_features
