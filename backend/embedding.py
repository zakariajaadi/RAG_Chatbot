from importlib import import_module

from langchain_core.embeddings import Embeddings

from config import RAGConfig


EMBEDDING_PROVIDERS = {
    "HuggingFaceEmbeddings":  "langchain_huggingface.HuggingFaceEmbeddings",
    "OpenAIEmbeddings":       "langchain_openai.OpenAIEmbeddings",
    "AzureOpenAIEmbeddings":  "langchain_openai.AzureOpenAIEmbeddings",
    "GoogleGenerativeAIEmbeddings": "langchain_google_genai.GoogleGenerativeAIEmbeddings"
}


def get_embedding_model(config: RAGConfig) -> Embeddings:
    
    source        = config.embedding_model.source
    source_config = dict(config.embedding_model.source_config)

    # If already an instance, return directly
    if isinstance(source, Embeddings):
        return source

    provider_path = EMBEDDING_PROVIDERS.get(source)
    if not provider_path:
        raise ValueError(f"Unknown embedding provider: '{source}'. Available: {list(EMBEDDING_PROVIDERS.keys())}")

    module_path, class_name = provider_path.rsplit(".", 1)
    try:
        # Dynamically import the embedding class from its module path (e.g. "langchain_openai.OpenAIEmbeddings")
        embedding_class = getattr(import_module(module_path), class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Cannot load '{provider_path}': {e}")

    try:
        # Unpack config params as keyword arguments to instantiate the embedding model
        return embedding_class(**source_config)
    except TypeError as e:
        raise TypeError(f"Invalid kwargs for '{source}': {e}")