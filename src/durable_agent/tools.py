from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from durable_agent.memory import MemoryStore
from durable_agent.rag import LocalRetriever


class Risk(str, Enum):
    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    risk: Risk
    parameters: dict[str, str]
    handler: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "risk": self.risk.value,
            "parameters": self.parameters,
        }


class ToolRegistry:
    def __init__(self, retriever: LocalRetriever | None = None, memory: MemoryStore | None = None) -> None:
        self.retriever = retriever or LocalRetriever()
        self.memory = memory or MemoryStore()
        self._tools = {
            "search_sources": ToolSpec(
                "search_sources", "Search local evidence and return citation-ready chunks.", Risk.READ,
                {"query": "string", "top_k": "integer"}, self.retriever.search,
            ),
            "read_source": ToolSpec(
                "read_source", "Read one allowed markdown source.", Risk.READ,
                {"filename": "string"}, self.retriever.read,
            ),
            "remember": ToolSpec(
                "remember", "Persist a useful note in long-term SQLite memory.", Risk.WRITE,
                {"content": "string", "kind": "string"}, self.memory.add,
            ),
        }

    def schemas(self, allowed: set[str] | None = None) -> list[dict[str, Any]]:
        names = allowed if allowed is not None else set(self._tools)
        return [self._tools[name].schema() for name in sorted(names) if name in self._tools]

    def execute(self, name: str, arguments: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
        started = time.perf_counter()
        if name not in allowed:
            return {"ok": False, "error": f"Tool {name} is not allowed for this task.", "duration_ms": 0}
        spec = self._tools.get(name)
        if spec is None:
            return {"ok": False, "error": f"Unknown tool: {name}", "duration_ms": 0}
        if spec.risk == Risk.DESTRUCTIVE:
            return {"ok": False, "error": "Destructive tools require explicit approval.", "duration_ms": 0}
        try:
            result = spec.handler(**arguments)
            return {
                "ok": True,
                "result": result,
                "duration_ms": int((time.perf_counter() - started) * 1000),
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "duration_ms": int((time.perf_counter() - started) * 1000),
            }
