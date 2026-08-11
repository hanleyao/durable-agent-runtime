from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal, TypedDict
from uuid import uuid4


TaskKind = Literal["research", "analysis", "report"]
TaskStatus = Literal["pending", "ready", "running", "done", "failed", "blocked", "skipped"]
RunPhase = Literal["created", "planning", "running", "evaluating", "done", "failed"]


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


@dataclass
class Task:
    id: str
    goal: str
    kind: TaskKind
    blocked_by: list[str] = field(default_factory=list)
    status: TaskStatus = "pending"
    attempts: int = 0
    max_attempts: int = 2
    revisions: int = 0
    max_revisions: int = 2
    result: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=now_iso)

    @classmethod
    def create(
        cls,
        goal: str,
        kind: TaskKind,
        *,
        task_id: str | None = None,
        blocked_by: list[str] | None = None,
        max_attempts: int = 2,
        max_revisions: int = 2,
    ) -> "Task":
        return cls(
            id=task_id or f"task_{uuid4().hex[:8]}",
            goal=goal,
            kind=kind,
            blocked_by=list(blocked_by or []),
            max_attempts=max(1, min(3, max_attempts)),
            max_revisions=max(0, min(3, max_revisions)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuntimeState(TypedDict, total=False):
    run_id: str
    goal: str
    phase: RunPhase
    tasks: dict[str, Task]
    current_task_id: str
    execution_result: dict[str, Any]
    evaluation: dict[str, Any]
    evaluation_count: int
    max_evaluations: int
    errors: list[str]
    events: list[dict[str, Any]]
    final_output: dict[str, Any]
