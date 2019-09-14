
from claf.data.reader.seq_cls import SeqClsReader
from claf.data.reader.cola import CoLAReader

from claf.data.reader.squad import SQuADReader

from claf.data.reader.wikisql import WikiSQLReader

from claf.data.reader.bert.seq_cls import SeqClsBertReader
from claf.data.reader.bert.glue.cola import CoLABertReader
from claf.data.reader.bert.glue.mrpc import MRPCBertReader
from claf.data.reader.bert.glue.mnli import MNLIBertReader
from claf.data.reader.bert.glue.qnli import QNLIBertReader
from claf.data.reader.bert.glue.qqp import QQPBertReader
from claf.data.reader.bert.glue.sst import SSTBertReader
from claf.data.reader.bert.glue.rte import RTEBertReader
from claf.data.reader.bert.glue.wnli import WNLIBertReader

from claf.data.reader.bert.regression import RegressionBertReader
from claf.data.reader.bert.glue.stsb import STSBBertReader

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
