from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from durable_agent.config import Settings
from durable_agent.models import now_iso


@dataclass(frozen=True)
class Message:
    id: int
    session_id: str
    role: str
    content: str
    route: str | None
    run_id: str | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "route": self.route,
            "run_id": self.run_id,
            "created_at": self.created_at,
        }


class ConversationStore:
    """Durable conversation history; context selection is handled separately."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or Settings.load().conversation_db).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    summary_through INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                    content TEXT NOT NULL,
                    route TEXT,
                    run_id TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session_id
                ON messages(session_id, id);
            """)
            connection.commit()

    def ensure_session(self, session_id: str, title: str = "") -> None:
        session_id = session_id.strip()
        if not session_id:
            raise ValueError("session_id cannot be empty.")
        timestamp = now_iso()
        with closing(self._connect()) as connection:
            connection.execute(
                """INSERT INTO sessions(id,title,created_at,updated_at) VALUES(?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at""",
                (session_id, title.strip()[:100] or session_id, timestamp, timestamp),
            )
            connection.commit()

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        route: str | None = None,
        run_id: str | None = None,
    ) -> Message:
        if role not in {"user", "assistant"}:
            raise ValueError("role must be user or assistant.")
        self.ensure_session(session_id, content if role == "user" else "")
        timestamp = now_iso()
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "INSERT INTO messages(session_id,role,content,route,run_id,created_at) VALUES(?,?,?,?,?,?)",
                (session_id, role, content, route, run_id, timestamp),
            )
            connection.execute("UPDATE sessions SET updated_at=? WHERE id=?", (timestamp, session_id))
            connection.commit()
            message_id = int(cursor.lastrowid)
        return Message(message_id, session_id, role, content, route, run_id, timestamp)

    @staticmethod
    def _message(row: sqlite3.Row) -> Message:
        return Message(
            int(row["id"]), row["session_id"], row["role"], row["content"],
            row["route"], row["run_id"], row["created_at"],
        )

    def history(self, session_id: str, limit: int = 50) -> list[Message]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, max(1, limit)),
            ).fetchall()
        return [self._message(row) for row in reversed(rows)]

    def context(self, session_id: str, recent_limit: int = 8) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            session = connection.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
            if session is None:
                return {"summary": "", "messages": [], "summary_through": 0}
            rows = connection.execute(
                "SELECT * FROM messages WHERE session_id=? AND id>? ORDER BY id DESC LIMIT ?",
                (session_id, int(session["summary_through"]), max(1, recent_limit)),
            ).fetchall()
        return {
            "summary": session["summary"],
            "messages": [self._message(row).to_dict() for row in reversed(rows)],
            "summary_through": int(session["summary_through"]),
        }

    def unsummarized(self, session_id: str) -> list[Message]:
        with closing(self._connect()) as connection:
            session = connection.execute("SELECT summary_through FROM sessions WHERE id=?", (session_id,)).fetchone()
            if session is None:
                return []
            rows = connection.execute(
                "SELECT * FROM messages WHERE session_id=? AND id>? ORDER BY id",
                (session_id, int(session["summary_through"])),
            ).fetchall()
        return [self._message(row) for row in rows]

    def update_summary(self, session_id: str, summary: str, through_message_id: int) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                "UPDATE sessions SET summary=?,summary_through=?,updated_at=? WHERE id=?",
                (summary.strip(), through_message_id, now_iso(), session_id),
            )
            connection.commit()

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT s.*, COUNT(m.id) AS message_count
                FROM sessions s LEFT JOIN messages m ON m.session_id=s.id
                GROUP BY s.id ORDER BY s.updated_at DESC LIMIT ?""",
                (max(1, limit),),
            ).fetchall()
        return [dict(row) for row in rows]
