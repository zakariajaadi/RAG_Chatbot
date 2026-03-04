from dataclasses import dataclass, field
from pathlib import Path
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.documents.compressor import BaseDocumentCompressor
from dotenv import load_dotenv
from omegaconf import OmegaConf

load_dotenv()


@dataclass
class LLMConfig:
    source:        str | BaseChatModel  = ""
    source_config: dict = field(default_factory=dict)


@dataclass
class EmbeddingConfig:
    source:        str | Embeddings = ""
    source_config: dict = field(default_factory=dict)


@dataclass
class RerankerConfig:
    source:        str | BaseDocumentCompressor = ""
    source_config: dict = field(default_factory=dict)


@dataclass
class VectorStoreConfig:
    persist_directory: str = "data/chroma_db"
    insertion_mode: str = "incremental"
    # Retrieval
    top_k:         int              = 10

    def __post_init__(self):
        # Resolve relative to project root
        base_dir = Path(__file__).resolve().parents[1]
        self.persist_directory = str(base_dir / self.persist_directory)

@dataclass
class DatabaseConfig:
    db_name: str = "record_manager_cache.sql"

    def __post_init__(self):
        base_dir = Path(__file__).resolve().parents[1]
        
        self.db_dir = base_dir / "data" / "record_manager"
        
        # Create directory if doesn't exist
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        # Final url for SQLAlchemy
        db_path = self.db_dir / self.db_name
        self.db_url = f"sqlite:///{db_path}"

@dataclass
class RAGConfig:
    """Centralized Config for RAG."""
    llm:             LLMConfig        = field(default_factory=LLMConfig)
    embedding_model: EmbeddingConfig  = field(default_factory=EmbeddingConfig)
    reranker:        RerankerConfig   = field(default_factory=RerankerConfig)
    vector_store:    VectorStoreConfig = field(default_factory=VectorStoreConfig)
    database:        DatabaseConfig    = field(default_factory=DatabaseConfig)
    # Chunking
    chunk_size:      int              = 500
    chunk_overlap:   int              = 50
    


def _load_source_config(cfg) -> dict:
    """Convert an OmegaConf source_config node to a native Python dict."""
    return OmegaConf.to_container(cfg.source_config, resolve=True)


def read_config() -> RAGConfig:
    base_dir    = Path(__file__).resolve().parents[1]
    config_path = base_dir / "config" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)

    return RAGConfig(
        llm=LLMConfig(
            source=cfg.RAG.llm.source,
            source_config=_load_source_config(cfg.RAG.llm),
        ),
        embedding_model=EmbeddingConfig(
            source=cfg.RAG.embedding_model.source,
            source_config=_load_source_config(cfg.RAG.embedding_model),
        ),
        reranker=RerankerConfig(
            source=cfg.RAG.reranker.source,
            source_config=_load_source_config(cfg.RAG.reranker),
        ),
        vector_store=VectorStoreConfig(
            persist_directory=cfg.RAG.vector_store.persist_directory,
            insertion_mode=cfg.RAG.vector_store.insertion_mode,
            top_k=cfg.RAG.vector_store.top_k
        ),
        database=DatabaseConfig(
            db_name=cfg.RAG.database.db_name
        ),
        chunk_size=cfg.RAG.chunk_size,
        chunk_overlap=cfg.RAG.chunk_overlap
    )


if __name__ == "__main__":
    conf = read_config()
    print(conf.llm.source)
    print(conf.embedding_model.source)
    print(conf.reranker.source)
    print(conf.vector_store.top_k)