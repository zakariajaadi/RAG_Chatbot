import os
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.retrievers import BaseRetriever

from langchain_core.retrievers import BaseRetriever

from config import RAGConfig
from llm import get_llm_model




class RAG:
    """
    RAG with reranking.
    """

    SYSTEM_PROMPT ="""
    You are a professional document analysis assistant. Your role is to provide accurate, concise responses based strictly on the provided context documents.

    CORE INSTRUCTIONS:
    - Respond only in the same language as the user's question
    - Use a professional, detailed writing style
    - Base responses exclusively on the provided context
    - Do not generate information not found in the context
    - Clearly distinguish between factual information and hypothetical scenarios

    RESPONSE GUIDELINES:
    1. If context is provided: Analyze and synthesize relevant information from the documents
    2. If no context is provided: Inform the user that context is required for an accurate response
    3. If asked for general knowledge: Request explicit permission before providing general information
    4. If the question is unclear: Ask for clarification or rephrasing

    CONTEXT VALIDATION:
    - Only use information directly supported by the provided documents
    - Ignore irrelevant or off-topic information
    - If the context doesn't contain sufficient information to answer the question, state this clearly

    Context Documents:
    {context}

    User Question:
    {question}

    Please provide your response following the above guidelines.
    """

    def __init__(self, config: RAGConfig, retriever: BaseRetriever):
        self.config   = config
        self._prompt  = ChatPromptTemplate.from_template(self.SYSTEM_PROMPT)
        self._llm     = get_llm_model(config)

        # Store the reranked retriever
        self.retriever = retriever
        
        # Build the execution chain
        self._chain   = self._build_chain()


    def query(self, question: str) -> str:
        """query rag with a question"""
        return self._chain.invoke(question)

    def stream(self, question: str):
        """Stream response."""
        return self._chain.stream(question)

    def get_relevant_documents(self, question: str) -> List[Document]:
        """
        Retrieve and rerank documents relevant to the given question.
        
        Useful for debugging retrieval quality, displaying sources to the user,
        or evaluating the pipeline without consuming LLM tokens.
        """
        return self.retriever.invoke(question)

    # Helpers 

    @staticmethod
    def _format_context(docs: List[Document]) -> str:
        return "\n\n".join(
            [f"[Source {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)]
        )

    def _build_chain(self):
        return (
            {
                "context":  self.retriever | self._format_context,
                "question": RunnablePassthrough(),
            }
            | self._prompt
            | self._llm
            | StrOutputParser()
        )

 


















