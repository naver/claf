
from claf.data.reader.seq_cls import SeqClsReader
from claf.data.reader.cola import CoLAReader

from claf.data.reader.squad import SQuADReader

from claf.data.reader.wikisql import WikiSQLReader

from claf.data.reader.bert.seq_cls import SeqClsBertReader
from claf.data.reader.bert.cola import CoLABertReader

from claf.data.reader.bert.squad import SQuADBertReader

from claf.data.reader.bert.tok_cls import TokClsBertReader
from claf.data.reader.bert.conll2003 import CoNLL2003BertReader


__all__ = ["SQuADReader", "SQuADBertReader", "WikiSQLReader",
           "SeqClsReader", "SeqClsBertReader", "TokClsBertReader",
           "CoLAReader", "CoLABertReader", "CoNLL2003BertReader"]  # for register
