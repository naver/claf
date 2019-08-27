
from claf.data.reader.seq_cls import SeqClsReader
from claf.data.reader.cola import CoLAReader

from claf.data.reader.squad import SQuADReader

from claf.data.reader.wikisql import WikiSQLReader

from claf.data.reader.bert.seq_cls import SeqClsBertReader
from claf.data.reader.bert.cola import CoLABertReader
from claf.data.reader.bert.mrpc import MRPCBertReader
from claf.data.reader.bert.mnli import MNLIBertReader
from claf.data.reader.bert.qnli import QNLIBertReader
from claf.data.reader.bert.qqp import QQPBertReader
from claf.data.reader.bert.sst import SSTBertReader
from claf.data.reader.bert.rte import RTEBertReader
from claf.data.reader.bert.wnli import WNLIBertReader

from claf.data.reader.bert.regression import RegressionBertReader
from claf.data.reader.bert.stsb import STSBBertReader

from claf.data.reader.bert.squad import SQuADBertReader

from claf.data.reader.bert.tok_cls import TokClsBertReader
from claf.data.reader.bert.conll2003 import CoNLL2003BertReader


# fmt: off

__all__ = [
    "RegressionBertReader", "STSBBertReader",

    "SeqClsReader", "CoLAReader",

    "SeqClsBertReader", "CoLABertReader", "MRPCBertReader", "MNLIBertReader", "QNLIBertReader",
    "QQPBertReader", "RTEBertReader", "SSTBertReader", "STSBBertReader", "WNLIBertReader",

    "SQuADReader",
    "SQuADBertReader",

    "TokClsBertReader", "CoNLL2003BertReader",

    "WikiSQLReader",
]

# fmt: on
