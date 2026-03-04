from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_core.documents import Document
from typing import List
from pathlib import Path


class DocumentLoader:
    """
    Loads documents from different sources.
    Supports : TXT, PDF
    Extensible to : WebBaseLoader, CSVLoader, etc.
    """

    @staticmethod
    def load_txt(path: str) -> List[Document]:
        loader = TextLoader(path, encoding="utf-8")
        return loader.load()

    @staticmethod
    def load_pdf(path: str) -> List[Document]:
        loader = PyPDFLoader(path)
        return loader.load()

    @staticmethod
    def load_csv(filepath: str) -> List[Document]:
        loader = CSVLoader(file_path=filepath, encoding="utf-8")
        return loader.load()


if __name__ == "__main__":

    # Test
    base_dir = Path(__file__).resolve().parents[1]
    file_path = base_dir / "content" / "fiches_clients_orange.txt"
    docs = DocumentLoader.load_txt(str(file_path))

    print(f"Number of documents loaded : {len(docs)}")
    print(f"Type : {type(docs[0])}")
    print(f"Metadata : {docs[0].metadata}")
    print(f"Content (excerpt) :\n{docs[0].page_content[:300]}")