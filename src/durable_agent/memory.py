from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from durable_agent.config import Settings
from durable_agent.models import now_iso
from durable_agent.rag import tokens


class MemoryStore:
    def __init__(self, database: str | Path | None = None) -> None:
        self.database = Path(database) if database else Settings.load().memory_db
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS memories(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )"""
            )

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.database, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def add(self, content: str, kind: str = "note", metadata: dict[str, Any] | None = None) -> int:
        if not content.strip():
            raise ValueError("Memory content is empty.")
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO memories(created_at,kind,content,metadata_json) VALUES(?,?,?,?)",
                (now_iso(), kind, content.strip(), json.dumps(metadata or {}, ensure_ascii=False)),
            )
            return int(cursor.lastrowid)

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM memories ORDER BY id DESC LIMIT 200").fetchall()
        query_tokens = tokens(query)
        ranked = []
        for row in rows:
            overlap = len(query_tokens & tokens(row["content"]))
            if overlap:
                ranked.append((overlap, row))
        ranked.sort(key=lambda item: (-item[0], -item[1]["id"]))
        return [
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "kind": row["kind"],
                "content": row["content"],
                "metadata": json.loads(row["metadata_json"]),
            }
            for _, row in ranked[:limit]
        ]
