"""
Conversation memory, in two interchangeable backends.

SQLite ties sessions to one machine's disk, which is fine for local development
and fatal the moment a second API replica exists. Postgres is selected by
setting DATABASE_URL; nothing else in the application changes.
"""

import json
import sqlite3
from pathlib import Path
from typing import Protocol

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class ConversationStore(Protocol):
    """What the pipeline needs from a memory backend."""

    def recent_turns(self, session_id: str, limit: int | None = None) -> list[dict]: ...

    def save_turn(
        self,
        session_id: str,
        question: str,
        rewritten_question: str,
        answer: str,
        source_path: str,
        citations: list[dict],
    ) -> None: ...

    def ping(self) -> bool: ...


CREATE_TABLE_SQLITE = """
CREATE TABLE IF NOT EXISTS conversation_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    question TEXT NOT NULL,
    rewritten_question TEXT NOT NULL,
    answer TEXT NOT NULL,
    source_path TEXT NOT NULL,
    citations_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

CREATE_TABLE_POSTGRES = """
CREATE TABLE IF NOT EXISTS conversation_turns (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    question TEXT NOT NULL,
    rewritten_question TEXT NOT NULL,
    answer TEXT NOT NULL,
    source_path TEXT NOT NULL,
    citations_json TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

# Every read is "the last N turns of one session", which this index serves
# directly instead of scanning the whole table as it grows.
CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_conversation_turns_session
ON conversation_turns (session_id, id DESC)
"""


class SQLiteMemory:
    """Single-process conversation history in a local file."""

    def __init__(self, database_path: Path | str | None = None):
        self.database_path = Path(database_path or settings.database_path)
        self._ensure_schema()

    def _connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path, timeout=10)

    def _ensure_schema(self) -> None:
        with self._connection() as connection:
            connection.execute(CREATE_TABLE_SQLITE)
            connection.execute(CREATE_INDEX)

    def recent_turns(self, session_id: str, limit: int | None = None) -> list[dict]:
        limit = limit or settings.history_turns
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT question, answer FROM conversation_turns
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        return [
            {"question": question, "answer": answer}
            for question, answer in reversed(rows)
        ]

    def save_turn(
        self,
        session_id: str,
        question: str,
        rewritten_question: str,
        answer: str,
        source_path: str,
        citations: list[dict],
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO conversation_turns
                (session_id, question, rewritten_question, answer, source_path,
                 citations_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    question,
                    rewritten_question,
                    answer,
                    source_path,
                    json.dumps(citations),
                ),
            )

    def ping(self) -> bool:
        try:
            with self._connection() as connection:
                connection.execute("SELECT 1")
            return True
        except Exception:
            logger.warning("SQLite memory is not reachable", exc_info=True)
            return False


class PostgresMemory:
    """Shared conversation history, safe across replicas."""

    def __init__(self, database_url: str | None = None):
        from psycopg_pool import ConnectionPool

        self.database_url = database_url or settings.database_url
        # A pool rather than a connection per call: opening a TCP connection and
        # authenticating on every turn would dominate the request's latency.
        self.pool = ConnectionPool(
            conninfo=self.database_url,
            min_size=settings.database_pool_min,
            max_size=settings.database_pool_max,
            open=True,
        )
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.pool.connection() as connection:
            connection.execute(CREATE_TABLE_POSTGRES)
            connection.execute(CREATE_INDEX)

    def recent_turns(self, session_id: str, limit: int | None = None) -> list[dict]:
        limit = limit or settings.history_turns
        with self.pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT question, answer FROM conversation_turns
                WHERE session_id = %s
                ORDER BY id DESC
                LIMIT %s
                """,
                (session_id, limit),
            ).fetchall()

        return [
            {"question": question, "answer": answer}
            for question, answer in reversed(rows)
        ]

    def save_turn(
        self,
        session_id: str,
        question: str,
        rewritten_question: str,
        answer: str,
        source_path: str,
        citations: list[dict],
    ) -> None:
        with self.pool.connection() as connection:
            connection.execute(
                """
                INSERT INTO conversation_turns
                (session_id, question, rewritten_question, answer, source_path,
                 citations_json)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    question,
                    rewritten_question,
                    answer,
                    source_path,
                    json.dumps(citations),
                ),
            )

    def ping(self) -> bool:
        try:
            with self.pool.connection() as connection:
                connection.execute("SELECT 1")
            return True
        except Exception:
            logger.warning("Postgres memory is not reachable", exc_info=True)
            return False

    def close(self) -> None:
        self.pool.close()


def create_memory() -> ConversationStore:
    """Pick the backend from DATABASE_URL."""
    if settings.uses_postgres:
        logger.info("using Postgres conversation memory")
        return PostgresMemory()

    logger.info(
        "using SQLite conversation memory (single replica only)",
        extra={"path": str(settings.database_path)},
    )
    return SQLiteMemory()


# Backwards-compatible alias: the pipeline and tests construct this by name, and
# it keeps SQLite's explicit-path constructor available for tests.
ConversationMemory = SQLiteMemory
