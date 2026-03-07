import os
from typing import List
from pathlib import Path
from loguru import logger
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_classic.indexes import SQLRecordManager, index
from langchain_core.documents import Document

from config import RAGConfig, read_config
from document_loader import DocumentLoader
from document_splitter import DocumentSplitter
from embedding import get_embedding_model
from vector_store_manager import VectorStoreManager

LOADER_DISPATCH = {
    ".pdf": DocumentLoader.load_pdf,
    ".txt": DocumentLoader.load_txt,
    ".csv": DocumentLoader.load_csv,
}
class Indexer:
    """
    Handles document indexing.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self._vsm = VectorStoreManager(
            embedding_model=get_embedding_model(config),
            persist_directory=config.vector_store.persist_directory,
        )
        # Cache to store SQLRecordManager instances by namespace.
        self._record_managers: dict[str, SQLRecordManager] = {}


    def ingest_directory(self, path: str, namespace: str = "default") -> None:
        """
        Scan directory and index files one by one with individual error handling.
        """
        # Filter directory entries to include only supported extensions
        files = [
            f for f in os.listdir(path)
            if Path(f).suffix.lower() in LOADER_DISPATCH
        ]
        if not files:
            logger.warning(f"No valid files found in {path}")
            return
        
        results = []
        # interate over files
        for filename in files:
            # Ingest a single file
            filepath = os.path.join(path, filename)
            res = self.ingest_file(filepath, namespace=namespace)
            results.append(res)
        
        errors = [r for r in results if r["status"] == "error"]
        logger.info(f"Directory processing finished. Success: {len(results)-len(errors)} | Errors: {len(errors)}")
    
    
    def ingest_file(self, path: str, namespace: str = "default") -> dict:
        """
        Ingest a single file with error handling.
        """
        try:
            chunks = self._load_and_split(path)
            if not chunks:
                logger.warning(f"File {Path(path).name} produced no chunks.")
                return {"file": Path(path).name, "chunks": 0, "status": "empty"}

            self._index_documents(chunks, namespace=namespace)
            return {"file": Path(path).name, "chunks": len(chunks), "status": "success"}

        except Exception as e:
            logger.exception(f"Error ingesting file {path}")
            return {"file": Path(path).name, "chunks": 0, "status": "error", "message": str(e)}
    

    
    # --- HELPERS ---

    def _split(self, docs: List[Document]) -> List[Document]:
        splitter = DocumentSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        return splitter.split(docs)
        
    def _load_and_split(self, path: str) -> List[Document]:
        """
        Load a single file and split it into chunks.
        Shared by ingest_directory and ingest_file.
        """
        # Extract filename and extension
        filename = Path(path).name
        ext = Path(path).suffix.lower()

        # Get suitable loader
        loader = LOADER_DISPATCH.get(ext)
        if not loader:
            raise ValueError(f"Unsupported file type: '{ext}'. Supported: {list(LOADER_DISPATCH.keys())}")
      
        # Load file as list of documents
        docs = loader(path)

        # Split documents into chunks (all file formats excpet CSV)
        chunks = docs if ext == ".csv" else self._split(docs)
        
        # Enrich chunks metadata with source file name
        for chunk in chunks:
            chunk.metadata["source_file"] = filename

        return chunks
    
    def _get_record_manager(self, namespace: str) -> SQLRecordManager:
        """Retrieve or initialize a SQLRecordManager for a specific namespace."""
        if namespace not in self._record_managers:
            rm = SQLRecordManager(namespace=namespace, db_url=self.config.database.db_url)
            rm.create_schema()
            self._record_managers[namespace] = rm
        return self._record_managers[namespace]
    
    def _index_documents(
        self,
        documents: List[Document],
        namespace: str = "default",
        insertion_mode: str = None,
    ) -> None:
        """
        Push vectors to the store with deduplication.
        """
        insertion_mode = insertion_mode or self.config.vector_store.insertion_mode
        
        # Remove complex metadata to ensure compatibility with the vector store 
        filtered_documents = filter_complex_metadata(documents)

        # Persistence: SQL Record Manager tracks hashes
        record_manager = self._get_record_manager(namespace)

        logger.info(f"Indexing {len(filtered_documents)} documents to {namespace}.")

        batch_size = 100
        for i in range(0, len(filtered_documents), batch_size):
            batch = filtered_documents[i : i + batch_size]
            
            logger.info(f"Indexing batch {i // batch_size + 1}...")

            # The index function orchestrates Add/Update/Delete
            indexing_output = index(
                batch,
                record_manager,
                self._vsm.get_store(),
                cleanup=insertion_mode,
                # Must match the key set in ingest_directory, for the record manager to be able to work as expected
                source_id_key="source_file", 
            )
            
            logger.info(f"Batch result: {indexing_output}")
    
if __name__ == "__main__":

    base_dir = Path(__file__).resolve().parents[1]
    dir_path = base_dir / "examples" 
    config = read_config()
    indexer = Indexer(config)
    indexer.ingest_directory(str(dir_path))
    print ("done")
