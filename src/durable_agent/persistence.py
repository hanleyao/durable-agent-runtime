from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver


def open_checkpointer(path: str | Path) -> tuple[SqliteSaver, sqlite3.Connection]:
    database = Path(path).expanduser().resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database, check_same_thread=False)
    serializer = JsonPlusSerializer(allowed_msgpack_modules=[("durable_agent.models", "Task")])
    saver = SqliteSaver(connection, serde=serializer)
    saver.setup()
    return saver, connection
