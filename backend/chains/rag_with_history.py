"""
rag_with_history.py

Assembles the full stateful RAG pipeline:
    1. condense_question  — rewrites follow-up questions as standalone questions
    2. rag_basic          — retrieves documents and generates an answer
    3. RunnableWithMessageHistory — persists conversation history per session

Input:  {"question": str}  +  configurable session_id
Output: str (answer)
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory

from chains.message_history import get_chat_message_history
from chains.condense_question import condense_question_chain
from config import RAGConfig
from chains.rag_basic import rag_basic_chain


def rag_with_history_chain(
    config: RAGConfig,
    llm: BaseChatModel,
    retriever: BaseRetriever,
) -> Runnable:
    """
    Stateful RAG chain with persistent memory.

    The question is first condensed with the chat history into a standalone
    question, which is then used for retrieval and answer generation.

    `RunnableWithMessageHistory` est un wrapper — il intercepte chaque `.invoke()` et fait trois choses automatiquement :

        1. Charge l'historique SQLite pour ce session_id
        2. Injecte l'historique dans le dict d'entrée sous la clé "chat_history"
        3. Après la réponse, sauvegarde le nouveau tour (question + réponse) en base

    """
    condense  = condense_question_chain(llm)
    answer    = rag_basic_chain(llm, retriever)

    # condense produces a standalone question (str) → fed directly into rag_basic
    core_chain = condense | answer

    return RunnableWithMessageHistory(
        core_chain,
        lambda session_id: get_chat_message_history(config, session_id),
        input_messages_key="question",
        history_messages_key="chat_history"
    )

