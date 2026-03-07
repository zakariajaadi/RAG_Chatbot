import os
import logging
from typing import List
from pathlib import Path
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_classic.indexes import SQLRecordManager, index
from langchain_core.documents import Document

from config import RAGConfig, read_config
from document_loader import DocumentLoader
from document_splitter import DocumentSplitter
from embedding import get_embedding_model
from vector_store_manager import VectorStoreManager

class Indexer:
    """
    Handles document indexing.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._vsm = VectorStoreManager(
            embedding_model=get_embedding_model(config),
            persist_directory=config.vector_store.persist_directory,
        )


    def ingest_directory(self, path: str, namespace: str = "default") -> None:
        """Scan directory and prepare documents for the indexation."""
        
        # Filter directory entries to include only .txt and .pdf files
        files = [f for f in os.listdir(path) if f.endswith((".txt", ".pdf",".csv"))]
        if not files:
            self.logger.warning(f"No valid files found in {path}")
            return
     
        all_chunks: List[Document] = []
        
        # Iterate over files
        for filename in files:
            filepath = os.path.join(path, filename)
            
            # Load file as list of documents using suitable loader
            if filename.endswith(".pdf"):
                docs = DocumentLoader.load_pdf(filepath)
            elif filename.endswith(".txt"):
                docs = DocumentLoader.load_txt(filepath)
            elif filename.endswith(".csv"):
                 docs = DocumentLoader.load_csv(filepath)
            
            # Split documents into chunks (all file formats excpet CSV)
            chunks = docs if filename.endswith(".csv") else self._split(docs)

            # Enrich chunks metadata with source file name
            for chunk in chunks:
                chunk.metadata["source_file"] = filename

            all_chunks.extend(chunks)
            self.logger.info(f"Processed {filename}: {len(chunks)} chunks created.")

        # Upsert chunks into vectore store 
        self.index_documents(all_chunks, namespace=namespace)


    def index_documents(
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
        record_manager = SQLRecordManager(
            namespace=namespace, 
            db_url=self.config.database.db_url
        )
        record_manager.create_schema()

        self.logger.info(f"Indexing {len(filtered_documents)} documents to {namespace}.")

        batch_size = 100
        for i in range(0, len(filtered_documents), batch_size):
            batch = filtered_documents[i : i + batch_size]
            
            self.logger.info(f"Indexing batch {i // batch_size + 1}...")

            # The index function orchestrates Add/Update/Delete
            indexing_output = index(
                batch,
                record_manager,
                self._vsm.get_store(),
                cleanup=insertion_mode,
                # Must match the key set in ingest_directory, for the record manager to be able to work as expected
                source_id_key="source_file", 
            )
            
            self.logger.info(f"Batch result: {indexing_output}")

    # --- HELPERS ---

    def _split(self, docs: List[Document]) -> List[Document]:
        splitter = DocumentSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        return splitter.split(docs)

if __name__ == "__main__":

    base_dir = Path(__file__).resolve().parents[1]
    dir_path = base_dir / "examples" 

    config = read_config()

    indexer = Indexer(config)
    indexer.ingest_directory(str(dir_path))
    print ("done")
