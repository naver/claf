
import logging
import os

from overrides import overrides

from claf.config.factory.tokens import make_all_tokenizers
from claf.config.utils import convert_config2dict
from claf.data.data_handler import CachePath, DataHandler
from claf.decorator import register

from claf.machine.base import Machine
from claf.machine.knowlege_base.docs import read_wiki_articles


logger = logging.getLogger(__name__)


@register("machine:open_qa")
class OpenQA(Machine):
    """
    Open-Domain Question Answer Machine (DrQA)

    DrQA is a system for reading comprehension applied to open-domain question answering.

    * Args:
        config: machine_config
    """

    def __init__(self, config):
        super(OpenQA, self).__init__(config)
        self.data_handler = DataHandler(CachePath.MACHINE / "open_qa")

        self.load()

    @overrides
    def load(self):
        # Tokenizers
        tokenizers_config = convert_config2dict(self.config.tokenizers)
        tokenizers = make_all_tokenizers(tokenizers_config)

        # Knowledge Base
        # - Wiki
        knowledge_base_config = self.config.knowledge_base
        self.docs, doc_name = self._load_knowledge_base(knowledge_base_config)

        # Reasoning
        # - Document Retrieval
        # - Reading Comprehension Experiment
        reasoning_config = self.config.reasoning

        self.document_retrieval = self._load_document_retrieval(
            reasoning_config.document_retrieval, tokenizers["word"], basename=doc_name
        )
        self.rc_experiment = self.make_module(reasoning_config.reading_comprehension)
        print("Ready ..! \n")

    def _load_knowledge_base(self, config):
        docs = read_wiki_articles(config.wiki)  # TODO: fix read whole wiki
        doc_name = f"{os.path.basename(config.wiki)}-{len(docs)}-articles"
        return docs, doc_name

    def _load_document_retrieval(self, config, word_tokenizer, basename="docs"):
        dir_path = f"doc-{config.type}-{config.name}-{word_tokenizer.cache_name}"
        doc_retrieval_path = os.path.join(dir_path, basename)

        config.params = {
            "texts": [doc.title for doc in self.docs],
            "word_tokenizer": word_tokenizer,
        }
        document_retrieval = self.make_module(config)

        doc_retrieval_path = self.data_handler.convert_cache_path(doc_retrieval_path)
        if doc_retrieval_path.exists():
            document_retrieval.load(doc_retrieval_path)
        else:
            print("Start Document Retrieval Indexing ...")
            document_retrieval.init()
            document_retrieval.save(doc_retrieval_path)  # Save Cache
        print("Completed!")
        return document_retrieval

    @overrides
    def __call__(self, question):
        result_docs = self.search_documents(question)
        print("-" * 50)
        print("Doc Scores:")
        for doc in result_docs:
            print(f" - {doc[1]} : {doc[2]}")
        print("-" * 50)

        passages = []
        for result_doc in result_docs:
            doc_index = result_doc[0]
            doc = self.docs[doc_index]
            passages.append(doc.text)

        answers = []
        for passage in passages:
            answer_text = self.machine_reading(passage, question)
            answers.append(answer_text)

        ranked_answers = sorted(answers, key=lambda x: x["score"], reverse=True)
        return ranked_answers

    def search_documents(self, question):
        return self.document_retrieval.get_closest(question)

    def machine_reading(self, context, question):
        raw_feature = {"context": context, "question": question}
        return self.rc_experiment.predict(raw_feature)
