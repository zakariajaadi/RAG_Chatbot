from config import read_config
from embedding import get_embedding_model
from vector_store_manager import VectorStoreManager
from retriever import get_retriever 
import logging
from rag import RAG
import warnings

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")





import logging
from vector_store_manager import VectorStoreManager


def main():
    # 1. Initialisation de la configuration et du logging
    config = read_config()    

    # 2. Initialisation du Store (Gestion du stockage)
    vsm = VectorStoreManager(
        embedding_model=get_embedding_model(config), 
        persist_directory=config.vector_store.persist_directory
    )

    # 3. Get retriever, vectore sotre index is load here (lazy loading)
    retriever = get_retriever(config, vsm)

    # Inject retriever in RAG
    rag_chain = RAG(config, retriever)

    # 6. Exécution d'une requête
    question = "c'est quoi une crise de bâtiment dans le monopoly ?"
    print(f"\nQuestion: {question}\n")
    

    #relevant_docs= rag_chain.get_relevant_documents(question)
    
    #print(f"Réponse :\n{relevant_docs[0]}")

    response = rag_chain.query(question)
    print(f"Réponse :\n{response}")


if __name__ == "__main__":
    main()
