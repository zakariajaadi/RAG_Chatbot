"""
rag_basic.py

Stateless RAG chain: retrieves relevant documents and generates an answer.
No history awareness — used as a building block by higher-level chains.

Input:  str (question)
Output: str (answer)
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough




ANSWER_PROMPT = """
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


def rag_basic_chain(llm: BaseChatModel, retriever: BaseRetriever) -> Runnable:
    """
    Stateless RAG chain: retrieve → format → answer.

    Input:  str (question)
    Output: str (answer)
    """
    prompt = ChatPromptTemplate.from_template(ANSWER_PROMPT)
    return (
        {
            "context":  retriever | _format_context,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def _format_context(docs: List[Document]) -> str:
    return "\n\n".join(
        [f"[Source {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)]
    )