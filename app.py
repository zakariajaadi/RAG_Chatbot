"""
app.py — Chainlit frontend

Run with:
    uv run chainlit run backend/app.py

Features:
    - Password authentication (reuses existing user_management.py)
    - Persistent conversation history (custom DB data layer)
    - Streaming responses token by token
    - Conversation memory via RunnableWithMessageHistory
"""

import sys
from pathlib import Path
from typing import Optional

import chainlit as cl
from chainlit.data.base import BaseDataLayer

from config import read_config
from backend.rag import RAG
from backend.data_layer import DBDataLayer
from backend.routers.auth.user_management import authenticate_user

# ---------------------------------------------------------------------------
# Initialization — runs once at startup
# ---------------------------------------------------------------------------

config = read_config()
rag    = RAG(config=config)
chain  = rag.get_chain(memory=True)


# ---------------------------------------------------------------------------
# Data layer — tells Chainlit where to persist conversations
# ---------------------------------------------------------------------------

@cl.data_layer
def get_data_layer() -> BaseDataLayer:
    return DBDataLayer(config.database.db_url)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    """
    Called by Chainlit on every login attempt.
    Reuses authenticate_user() from user_management.py — bcrypt + SQLite.
    Returns a cl.User on success, None on failure.
    """
    user = authenticate_user(config.database.db_url, username, password)
    if user:
        return cl.User(identifier=username, metadata={"email": username})
    return None


# ---------------------------------------------------------------------------
# Chat lifecycle
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def start():
    """Called when a new conversation starts."""
    # Use Chainlit's thread_id as session_id for LangChain memory
    cl.user_session.set("session_id", cl.context.session.thread_id)


@cl.on_chat_resume
async def resume(thread):
    """Called when the user resumes a past conversation from the sidebar."""
    cl.user_session.set("session_id", thread["id"])


@cl.on_message
async def main(message: cl.Message):
    """Called on every user message."""
    session_id = cl.user_session.get("session_id")

    # Create empty message to stream into
    msg = cl.Message(content="")
    await msg.send()

    async for chunk in chain.astream(
        {"question": message.content},
        config={"configurable": {"session_id": session_id}},
    ):
        await msg.stream_token(chunk)

    await msg.update()