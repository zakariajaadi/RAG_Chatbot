from importlib import import_module

from langchain_core.documents.compressor import BaseDocumentCompressor

from config import RAGConfig


RERANKER_PROVIDERS = {
    "CohereRerank":       "langchain_cohere.CohereRerank",
    "FlashrankRerank":    "langchain_community.document_compressors.flashrank_rerank.FlashrankRerank",
    "HuggingFaceCrossEncoder": "langchain_community.document_compressors.cross_encoder_rerank.CrossEncoderReranker",
}


def get_reranker(config: RAGConfig) -> BaseDocumentCompressor:
    
    source        = config.reranker.source
    source_config = dict(config.reranker.source_config)

    # If already an instance, return directly
    if isinstance(source, BaseDocumentCompressor):
        return source

    provider_path = RERANKER_PROVIDERS.get(source)
    if not provider_path:
        raise ValueError(f"Unknown reranker provider: '{source}'. Available: {list(RERANKER_PROVIDERS.keys())}")

    module_path, class_name = provider_path.rsplit(".", 1)
    try:
        # Dynamically import the reranker class from its module path (e.g. "langchain_cohere.CohereRerank")
        reranker_class = getattr(import_module(module_path), class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Cannot load '{provider_path}': {e}")

    try:
        # Unpack config params as keyword arguments to instantiate the reranker
        return reranker_class(**source_config)
    except TypeError as e:
        raise TypeError(f"Invalid kwargs for '{source}': {e}")