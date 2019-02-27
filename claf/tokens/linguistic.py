class POSTag:
    """
        Universal POS tags expends by spacy
        (https://spacy.io/api/annotation#section-pos-tagging)
    """

    classes = [
        "ADJ",  # adjectives
        "ADP",  # adpositions (prepositions and postpositions)
        "ADV",  # adverbs
        "AUX",  # auxiliary (spacy)
        "CONJ",  # conjunctions
        "CCONJ",  # coordinating conjunction (spacy)
        "DET",  # determiners
        "INTJ",  # interjection (spacy)
        "NOUN",  # nouns (common and proper)
        "NUM",  # cardinal numbers
        "PART",  # particles or other function words  (spacy)
        "PRON",  # pronouns
        "PROPN",  # proper noun
        "PUNCT",  # punctuation
        "SCONJ",  # subordinating conjunction
        "SYM",  # symbol
        "VERB",  # verbs (all tenses and modes)
        "X",  # other: foreign words, typos, abbreviations
        "SPACE",  # space
    ]


class NER:
    """
        Named Entity Recognition

        Models trained on the OntoNotes 5 corpus support
        the following entity types:
        (https://spacy.io/api/annotation#section-dependency-parsing)
    """

    classes = [
        "NONE",  # None
        "PERSON",  # People, including fictional.
        "NORP",  # Nationalities or religious or political groups.
        "FAC",  # Buildings, airports, highways, bridges, etc.
        "ORG",  # Companies, agencies, institutions, etc.
        "GPE",  # Countries, cities, states.
        "LOC",  # Non-GPE locations, mountain ranges, bodies of water.
        "PRODUCT",  # Objects, vehicles, foods, etc. (Not services.)
        "EVENT",  # Named hurricanes, battles, wars, sports events, etc.
        "WORK_OF_ART",  # Titles of books, songs, etc.
        "LAW",  # Named documents made into laws.
        "LANGUAGE",  # Any named language.
        "DATE",  # Absolute or relative dates or periods.
        "TIME",  # Times smaller than a day.
        "PERCENT",  # Percentage, including "%".
        "MONEY",  # Monetary values, including unit.
        "QUANTITY",  # Measurements, as of weight or distance.
        "ORDINAL",  # "first", "second", etc.
        "CARDINAL",  # Numerals that do not fall under another type.
    ]
