from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.retrievers import BaseRetriever
from vector_store_manager import VectorStoreManager
from reranker import get_reranker
from config import RAGConfig 
   


def get_retriever(config: RAGConfig, vector_store: VectorStoreManager) -> BaseRetriever:
    """
    Factory function to build a retriever (VectorStore + Reranker).
    """
    # Base search (Chroma)
    base_retriever = vector_store.as_retriever(k=config.vector_store.top_k)
    
    # get Reranker
    compressor = get_reranker(config)
    
    # Combine both
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )