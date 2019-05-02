#nsml: claf/claf:latest

from distutils.core import setup

# from claf import __version__ as VERSION


setup(
    name="nsml: reqsoning-qa",
    version="1.0",
    description="ns-ml",
    install_requires=[
        "numpy", "torch>=0.4.1",
        "konlpy", "nltk", "spacy",  # Tokenizer
        "babel", "records",  # WikiSQL
        "h5py", "jsbeautifier", "overrides", "requests", "gensim", "tqdm", "tensorboardX",  # Utils
        "pycm", "seqeval",
    ],
)
