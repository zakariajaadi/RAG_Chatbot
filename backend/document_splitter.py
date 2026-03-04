from langchain_core.callbacks import file
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from document_loader import DocumentLoader      
from pathlib import Path


class DocumentSplitter:
    """
    Split documents into chunks optimized for retrieval.

    chunk_size    : Maximum chunk size in characters
    chunk_overlap : Chunk overlap for context preservation at boundaries
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter     = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split(self, documents: List[Document]) -> List[Document]:
        chunks = self._splitter.split_documents(documents)
        # enrich chunks metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"]   = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
        return chunks

if __name__ == "__main__":

   pass