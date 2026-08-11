from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from durable_agent.config import Settings
from durable_agent.models import now_iso


class TraceLogger:
    def __init__(self, run_id: str, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        safe_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in run_id)
        self.path = self.settings.trace_dir / f"{safe_id}.jsonl"

    def event(self, event_type: str, **data: Any) -> None:
        payload = {"timestamp": now_iso(), "type": event_type, **data}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
