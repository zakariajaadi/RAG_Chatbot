"""
rag.py

RAG pipeline entry point. Thin orchestration class that delegates
chain construction to dedicated modules.

Two modes:
  - Stateless (default): each query is independent, no history
  - Stateful:            persistent memory per session_id,
                         question is condensed before retrieval

"""

from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from config import RAGConfig
from embedding import get_embedding_model
from llm import get_llm_model
from retriever import get_retriever
from vector_store_manager import VectorStoreManager
from chains.rag_basic import rag_basic_chain
from chains.rag_with_history import rag_with_history_chain


class RAG:
    """
    - RAG pipeline with reranking and optional persistent memory.
    - Self-contained: builds its own vector store and retriever from config.
    - Chains are built lazily — only when get_chain() is called.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.llm    = get_llm_model(config)

        vsm = VectorStoreManager(
            embedding_model=get_embedding_model(config),
            persist_directory=config.vector_store.persist_directory,
        )
        self.retriever = get_retriever(config, vsm)

    def get_chain(self, memory: bool = False) -> Runnable:
        """
        Returns the appropriate chain based on the memory flag.

        Args:
            memory: If False (default), returns a stateless chain (str → str).
                    If True, returns a stateful chain with persistent memory
                    ({question: str} → str), requires session_id at invoke time.
        """
        if memory:
            return rag_with_history_chain(self.config, self.llm, self.retriever)
        return rag_basic_chain(self.llm, self.retriever)

    def get_relevant_documents(self, question: str) -> List[Document]:
        """Retrieve and rerank documents. Useful for debugging without LLM tokens."""
        return self.retriever.invoke(question)