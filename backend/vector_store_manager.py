from loguru import logger 
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


class VectorStoreManager:
    """
    Manages the creation and loading of a Chroma index.
    """

    def __init__(self, embedding_model: Embeddings, persist_directory: str = "data/chroma_db"):
        self.embeddings = embedding_model
        self.persist_directory = persist_directory
        self.vectorstore = None

    def get_store(self) -> Chroma:
        """
        Returns the vector store instance from t
        If it's not loaded, it initializes it (creating the directory if needed).
        """
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        return self.vectorstore

    def load(self) -> "VectorStoreManager":
        """Load an existing Chroma index."""
        self.get_store()
        logger.info(f"Index loaded from {self.persist_directory}")
        return self

    def as_retriever(self, k: int = 4):
        """Return a standard LangChain retriever."""
        store = self.get_store()
        return store.as_retriever(search_kwargs={"k": k})

