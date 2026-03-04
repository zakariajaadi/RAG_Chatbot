"""
Manages persistent conversation history per session.
Uses SQLite via SQLChatMessageHistory with a custom timestamped model.
"""

from datetime import datetime

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.chat_message_histories.sql import DefaultMessageConverter
from sqlalchemy import Column, DateTime, Integer, Text
from sqlalchemy.ext.asyncio import create_async_engine

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

from config import RAGConfig


TABLE_NAME = "message_history"


def get_chat_message_history(config: RAGConfig, session_id: str) -> SQLChatMessageHistory:
    """
    Returns a persistent chat history store for a given session.
    Compatible with RunnableWithMessageHistory's session factory signature.

    SQLChatMessageHistory: 
        - Is the class responsible for reading/writing messages to a SQL table.
        - It natively serializes message objects (e.g., HumanMessage, AIMessage) into JSON format within the message column.
    """
    engine = create_async_engine(config.database.db_url_async)
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=engine,
        table_name=TABLE_NAME,
       # Add timestamp column via custom converter (LangChain default: id, session_id, message)
        custom_message_converter=TimestampedMessageConverter(TABLE_NAME),
    )


class TimestampedMessageConverter(DefaultMessageConverter):
    """
    Extends the default LangChain message converter with a timestamp column.
    Enables ordering and filtering by date (sliding window, audit, etc.).
    
    This class : 
        - Inherits DefaultMessageConverter to keep all serialization / deserilaization logic of messages 
        - Only replaces SQLAlchemy model by a custom model with timestamp columns
    """

    def __init__(self, table_name: str):
        self.model_class = _create_message_model(table_name, declarative_base())


def _create_message_model(table_name: str, dynamic_base):
    class Message(dynamic_base):
        __tablename__ = table_name
        id         = Column(Integer, primary_key=True)
        timestamp  = Column(DateTime, default=datetime.utcnow)
        session_id = Column(Text)
        message    = Column(Text)

    return Message