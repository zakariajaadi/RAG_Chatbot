from importlib import import_module
from typing import List

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel


# Registry mapping provider names to their import paths
LLM_PROVIDERS = {
    "ChatOpenAI":      "langchain_openai.ChatOpenAI",
    "AzureChatOpenAI": "langchain_openai.AzureChatOpenAI",
    "ChatGoogleGenerativeAI":   "langchain_google_genai.ChatGoogleGenerativeAI",
    "ChatAnthropic":   "langchain_anthropic.ChatAnthropic",
    "ChatMistralAI":   "langchain_mistralai.ChatMistralAI",
}


def get_llm_model(config, callbacks: List[BaseCallbackHandler] = []) -> BaseChatModel:
    source = config.llm.source
    source_config = config.llm.source_config

    # If already an instance, return directly
    if isinstance(source, BaseChatModel):
        return source
    
    # Look up the provider class path
    provider_path = LLM_PROVIDERS.get(source)
    if not provider_path:
        raise ValueError(f"Unknown LLM provider: '{source}'. Available: {list(LLM_PROVIDERS.keys())}")

    
    # Dynamically import the LLM class from its module path (e.g. "langchain_openai.ChatOpenAI")
    module_path, class_name = provider_path.rsplit(".", 1)
    llm_class = getattr(import_module(module_path), class_name)
    

    # Add callbacks to the source config
    source_config["callbacks"] = callbacks

    # Try to instantiate the LLM class with the source config
    try:
        return llm_class(**source_config)
    except TypeError as e:
        raise TypeError(f"Invalid kwargs for '{source}': {e}")