"""
data_layer.py

Custom Chainlit data layer backed by SQLite (via our Database class).
Implements the BaseDataLayer interface so Chainlit can persist:
  - Users
  - Conversations (threads)
  - Messages (steps)

Chainlit calls these methods automatically — no need to call them manually.

Usage in app.py:
    from data_layer import DBDataLayer
    cl.data_layer = DBDataLayer(db_url)
"""

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional
from pathlib import Path

from chainlit.data.base import BaseDataLayer
from chainlit.types import (
    Feedback,
    PageInfo,
    PaginatedResponse,
    ThreadDict,
    ThreadFilter,
)
from chainlit.user import PersistedUser, User

from database import Database


CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS cl_users (
    id TEXT PRIMARY KEY,
    identifier TEXT UNIQUE NOT NULL,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cl_threads (
    id TEXT PRIMARY KEY,
    name TEXT,
    user_id TEXT,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES cl_users(id)
);

CREATE TABLE IF NOT EXISTS cl_steps (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    type TEXT NOT NULL,
    name TEXT,
    input TEXT,
    output TEXT,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (thread_id) REFERENCES cl_threads(id)
);
"""


class DBDataLayer(BaseDataLayer):
    """
    Chainlit data layer backed by a Database.
    Persists users, threads (conversations), and steps (messages).
    """

    def __init__(self, db_url: str):
        self.db_url = db_url
        self._init_tables()

    def _init_tables(self) -> None:
        """Create tables if they don't exist."""
        with Database(self.db_url) as db:
            for statement in CREATE_TABLES_SQL.strip().split(";"):
                statement = statement.strip()
                if statement:
                    db.execute(statement)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------ #
    # Users
    # ------------------------------------------------------------------ #

    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        with Database(self.db_url) as db:
            row = db.fetchone(
                "SELECT id, identifier, metadata, created_at FROM cl_users WHERE identifier = ?",
                (identifier,),
            )
        if not row:
            return None
        return PersistedUser(
            id=row[0],
            identifier=row[1],
            metadata=json.loads(row[2]),
            createdAt=row[3],
        )

    async def create_user(self, user: User) -> Optional[PersistedUser]:
        import uuid
        user_id = str(uuid.uuid4())
        now = self._now()
        with Database(self.db_url) as db:
            db.execute(
                "INSERT OR IGNORE INTO cl_users (id, identifier, metadata, created_at) VALUES (?, ?, ?, ?)",
                (user_id, user.identifier, json.dumps(user.metadata or {}), now),
            )
        return PersistedUser(
            id=user_id,
            identifier=user.identifier,
            metadata=user.metadata or {},
            createdAt=now,
        )

    # ------------------------------------------------------------------ #
    # Threads (conversations)
    # ------------------------------------------------------------------ #

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Chainlit uses update_thread as an 'upsert' mechanism. 
        If the thread doesn't exist, we must create it.
        """
        with Database(self.db_url) as db:
            # 1. Create the thread if it doesn't exist (Initial Insert)
            db.execute(
                "INSERT OR IGNORE INTO cl_threads (id, name, user_id, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    thread_id,
                    name or "New conversation",
                    user_id,
                    json.dumps(metadata or {}),
                    self._now(),
                ),
            )

            # 2. Update existing fields if new values are provided
            if name:
                db.execute(
                    "UPDATE cl_threads SET name = ? WHERE id = ?",
                    (name, thread_id),
                )
            if user_id:
                db.execute(
                    "UPDATE cl_threads SET user_id = ? WHERE id = ?",
                    (user_id, thread_id),
                )
            if metadata:
                db.execute(
                    "UPDATE cl_threads SET metadata = ? WHERE id = ?",
                    (json.dumps(metadata), thread_id),
                )

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        with Database(self.db_url) as db:
            # We join with cl_users to get the identifier (e.g., username/email)
            row = db.fetchone(
                """
                SELECT t.id, t.name, t.user_id, t.metadata, t.created_at, u.identifier
                FROM cl_threads t
                LEFT JOIN cl_users u ON t.user_id = u.id
                WHERE t.id = ?
                """,
                (thread_id,),
            )
            
            if not row:
                return None
                
            steps = db.fetchall(
                "SELECT id, thread_id, type, name, input, output, metadata, created_at FROM cl_steps WHERE thread_id = ? ORDER BY created_at ASC",
                (thread_id,),
            )

        return {
            "id": row[0],
            "name": row[1],
            "userId": row[2],
            "metadata": json.loads(row[3]),
            "createdAt": row[4],
            "userIdentifier": row[5], 
            "steps": [
                {
                    "id": s[0],
                    "threadId": s[1],
                    "type": s[2],
                    "name": s[3],
                    "input": s[4],
                    "output": s[5],
                    "metadata": json.loads(s[6]),
                    "createdAt": s[7],
                }
                for s in steps
            ],
        }
    async def get_thread_author(self, thread_id: str) -> str:
        with Database(self.db_url) as db:
            row = db.fetchone(
                "SELECT user_id FROM cl_threads WHERE id = ?",
                (thread_id,),
            )
        return row[0] if row else ""

    async def delete_thread(self, thread_id: str) -> None:
        with Database(self.db_url) as db:
            db.execute("DELETE FROM cl_steps WHERE thread_id = ?", (thread_id,))
            db.execute("DELETE FROM cl_threads WHERE id = ?", (thread_id,))

    async def list_threads(
        self, pagination: any, filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        with Database(self.db_url) as db:
            rows = db.fetchall(
                "SELECT id, name, user_id, metadata, created_at FROM cl_threads WHERE user_id = ? ORDER BY created_at DESC",
                (filters.userId,),
            )
        
        threads = [
            {
                "id": r[0],
                "name": r[1],
                "userId": r[2],
                "metadata": json.loads(r[3]),
                "createdAt": r[4],
            }
            for r in rows
        ]
        return PaginatedResponse(
            data=threads,
            pageInfo=PageInfo(hasNextPage=False, startCursor=None, endCursor=None),
        )

    # ------------------------------------------------------------------ #
    # Steps (messages)
    # ------------------------------------------------------------------ #

    async def create_step(self, step_dict: Dict) -> None:
        with Database(self.db_url) as db:
            db.execute(
                "INSERT OR IGNORE INTO cl_steps (id, thread_id, type, name, input, output, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    step_dict.get("id", ""),
                    step_dict.get("threadId", ""),
                    step_dict.get("type", ""),
                    step_dict.get("name", ""),
                    step_dict.get("input", ""),
                    step_dict.get("output", ""),
                    json.dumps(step_dict.get("metadata", {})),
                    step_dict.get("createdAt", self._now()),
                ),
            )

    async def update_step(self, step_dict: Dict) -> None:
        with Database(self.db_url) as db:
            db.execute(
                "UPDATE cl_steps SET output = ?, metadata = ? WHERE id = ?",
                (
                    step_dict.get("output", ""),
                    json.dumps(step_dict.get("metadata", {})),
                    step_dict.get("id", ""),
                ),
            )

    async def delete_step(self, step_id: str) -> None:
        with Database(self.db_url) as db:
            db.execute("DELETE FROM cl_steps WHERE id = ?", (step_id,))

    # ------------------------------------------------------------------ #
    # Feedback (Optional, required by the interface)
    # ------------------------------------------------------------------ #

    async def upsert_feedback(self, feedback: Feedback) -> str:
        return ""

    async def delete_feedback(self, feedback_id: str) -> bool:
        return True

    async def get_element(self, thread_id: str, element_id: str):
        return None

    async def create_element(self, element) -> None:
        pass

    async def update_element(self, element) -> None:
        pass

    async def delete_element(self, element_id: str, thread_id: Optional[str] = None) -> None:
        pass

    async def build_debug_url(self) -> str:
        return ""

    async def close(self) -> None:
        pass

    async def get_favorite_steps(self, thread_id: str):
        return []