from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
from typing import Any, Callable
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from durable_agent.agent import TaskAgent
from durable_agent.config import Settings
from durable_agent.evaluator import QualityEvaluator
from durable_agent.llm import OpenAICompatibleClient
from durable_agent.memory import MemoryStore
from durable_agent.models import RuntimeState, Task
from durable_agent.persistence import open_checkpointer
from durable_agent.planner import Planner
from durable_agent.trace import TraceLogger


Progress = Callable[[str], None]
TERMINAL = {"done", "failed", "blocked", "skipped"}


def append_event(state: RuntimeState, event_type: str, **data: Any) -> list[dict[str, Any]]:
    return [*state.get("events", []), {"type": event_type, **data}]


def completed_results(tasks: dict[str, Task]) -> dict[str, dict[str, Any]]:
    return {task_id: task.result or {} for task_id, task in tasks.items() if task.status == "done"}


def merge_replanned_tasks(
    previous: dict[str, Task],
    replanned: dict[str, Task],
    feedback: str,
) -> tuple[dict[str, Task], list[str]]:
    """Reuse only completed tasks whose stable identity and semantics did not change."""
    reused: list[str] = []
    for task_id, task in replanned.items():
        old = previous.get(task_id)
        if (
            old and old.status == "done" and old.kind == task.kind
            and old.goal.strip() == task.goal.strip() and old.blocked_by == task.blocked_by
        ):
            task.status = "done"
            task.result = old.result
            task.attempts = old.attempts
            task.revisions = old.revisions
            reused.append(task_id)
        else:
            task.metadata["evaluation_feedback"] = feedback
            task.metadata["replanned"] = True
    return replanned, reused


def build_graph(
    planner: Planner,
    agent: TaskAgent,
    evaluator: QualityEvaluator,
    trace: TraceLogger,
    *,
    checkpointer: Any | None = None,
    progress: Progress | None = None,
    pause_before_node: str | None = None,
    pause_seconds: float = 0.0,
    max_runtime_seconds: float = 480.0,
):
    invocation_started = time.monotonic()

    def budget_exhausted() -> bool:
        return max_runtime_seconds > 0 and time.monotonic() - invocation_started >= max_runtime_seconds

    def pause(node: str) -> None:
        if pause_before_node != node:
            return
        trace.event("fault_window", node=node, pause_seconds=pause_seconds)
        if progress:
            progress(f"[fault-window] before={node} seconds={pause_seconds}")
        time.sleep(max(0.0, pause_seconds))

    def plan_node(state: RuntimeState) -> RuntimeState:
        pause("plan")
        if state.get("tasks"):
            return {"phase": "planning"}
        tasks, repairs, reasoning = planner.plan(state["goal"])
        trace.event("plan_created", tasks=list(tasks), repairs=repairs, reasoning=reasoning)
        if progress:
            progress(f"[plan] tasks={len(tasks)} repairs={len(repairs)}")
        return {
            "tasks": tasks,
            "phase": "planning",
            "events": append_event(state, "plan_created", tasks=list(tasks), repairs=repairs),
        }

    def schedule_node(state: RuntimeState) -> RuntimeState:
        pause("schedule")
        if budget_exhausted():
            return {
                "phase": "failed",
                "errors": [*state.get("errors", []), "Runtime time budget exhausted."],
                "current_task_id": "",
            }
        tasks = state["tasks"]
        for task in tasks.values():
            if task.status not in {"pending", "ready"}:
                continue
            dependencies = [tasks[item] for item in task.blocked_by]
            if any(item.status in {"failed", "blocked", "skipped"} for item in dependencies):
                task.status = "blocked"
                task.error = "A dependency did not complete successfully."
            elif all(item.status == "done" for item in dependencies):
                task.status = "ready"
        ready = next((task for task in tasks.values() if task.status == "ready"), None)
        if ready:
            ready.status = "running"
            ready.attempts += 1
            if progress:
                revision = f" revision={ready.revisions}/{ready.max_revisions}" if ready.revisions else ""
                progress(f"[task] {ready.id} ({ready.kind}) attempt={ready.attempts}/{ready.max_attempts}{revision}")
            trace.event("task_dispatched", task_id=ready.id, kind=ready.kind, attempt=ready.attempts)
            return {
                "phase": "running",
                "current_task_id": ready.id,
                "tasks": tasks,
                "events": append_event(state, "task_dispatched", task_id=ready.id),
            }
        if all(task.status in TERMINAL for task in tasks.values()):
            return {"phase": "evaluating", "current_task_id": "", "tasks": tasks}
        return {
            "phase": "failed",
            "errors": [*state.get("errors", []), "Scheduler deadlock: no ready task."],
            "current_task_id": "",
        }

    def route_schedule(state: RuntimeState) -> str:
        if state.get("current_task_id"):
            return "execute"
        return "evaluate" if state.get("phase") == "evaluating" else "finalize"

    def execute_node(state: RuntimeState) -> RuntimeState:
        pause("execute")
        task = state["tasks"][state["current_task_id"]]
        result = agent.execute(state["goal"], task, completed_results(state["tasks"]))
        trace.event("task_executed", task_id=task.id, ok=result.get("ok"), error=result.get("error"))
        return {"execution_result": result}

    def handle_node(state: RuntimeState) -> RuntimeState:
        pause("handle")
        task = state["tasks"][state["current_task_id"]]
        execution = state.get("execution_result", {})
        if execution.get("ok"):
            task.status = "done"
            task.result = execution.get("result") or {}
            task.error = None
            outcome = "done"
        else:
            task.error = str(execution.get("error") or "Task failed.")
            if execution.get("retryable", True) is not False and task.attempts < task.max_attempts:
                task.status = "ready"
                outcome = "retry"
            else:
                task.status = "failed"
                outcome = "failed"
        if progress:
            progress(f"[result] {task.id} -> {outcome}")
        trace.event("task_result", task_id=task.id, outcome=outcome, error=task.error)
        return {
            "tasks": state["tasks"],
            "current_task_id": "",
            "execution_result": {},
            "events": append_event(state, "task_result", task_id=task.id, outcome=outcome),
        }

    def evaluate_node(state: RuntimeState) -> RuntimeState:
        pause("evaluate")
        tasks = state["tasks"]
        results = completed_results(tasks)
        report_result = next(
            (task.result or {} for task in reversed(list(tasks.values())) if task.kind == "report" and task.result),
            {},
        )
        evidence = [item for value in results.values() for item in value.get("evidence", []) if isinstance(item, dict)]
        citations = [item for value in results.values() for item in value.get("citations", []) if isinstance(item, dict)]
        used = [item for item in report_result.get("used_citations", []) if isinstance(item, str)]
        relevant = [task for task in tasks.values() if task.kind in {"research", "analysis", "report"}]
        report = evaluator.evaluate({
            "goal": state["goal"],
            "answer": report_result.get("report") or report_result.get("answer") or "",
            "evidence": evidence,
            "citations": citations,
            "used_citations": used,
            "limitations": report_result.get("limitations", []),
            "confidence": report_result.get("confidence"),
            "completed_tasks": sum(task.status == "done" for task in relevant),
            "total_tasks": len(relevant),
            "failed_tasks": [task.id for task in relevant if task.status in {"failed", "blocked"}],
        })
        evaluation_count = state.get("evaluation_count", 0) + 1
        trace.event("evaluation", evaluation=report.to_dict(), count=evaluation_count)
        if progress:
            progress(f"[evaluate] score={report.overall_score:.2f} action={report.action}")
        return {
            "evaluation": report.to_dict(),
            "evaluation_count": evaluation_count,
            "phase": "done" if report.passed else "evaluating",
            "events": append_event(state, "evaluation", action=report.action, score=report.overall_score),
        }

    def route_evaluate(state: RuntimeState) -> str:
        evaluation = state["evaluation"]
        if evaluation.get("passed"):
            return "finalize"
        if (
            evaluation.get("action") == "abort"
            or state.get("evaluation_count", 0) >= state.get("max_evaluations", 3)
            or budget_exhausted()
        ):
            return "finalize"
        return "repair"

    def repair_node(state: RuntimeState) -> RuntimeState:
        pause("repair")
        tasks = state["tasks"]
        action = state["evaluation"]["action"]
        instruction = state["evaluation"].get("revision_instruction", "")
        if action == "replan":
            replanned, repairs, reasoning = planner.replan(state["goal"], tasks, state["evaluation"])
            merged, reused = merge_replanned_tasks(tasks, replanned, instruction)
            replan_count = state.get("replan_count", 0) + 1
            history = [*state.get("replan_history", []), {
                "generation": replan_count,
                "reasoning": reasoning,
                "repairs": repairs,
                "reused_tasks": reused,
                "task_ids": list(merged),
            }]
            trace.event(
                "plan_replanned", generation=replan_count, reasoning=reasoning,
                repairs=repairs, reused_tasks=reused, tasks=list(merged),
            )
            if progress:
                progress(f"[replan] generation={replan_count} tasks={len(merged)} reused={len(reused)} repairs={len(repairs)}")
            return {
                "tasks": merged,
                "phase": "running",
                "current_task_id": "",
                "replan_count": replan_count,
                "replan_history": history,
            }
        targets = [task for task in tasks.values() if task.kind == "report"]
        for task in targets:
            if task.revisions < task.max_revisions:
                task.revisions += 1
                task.attempts = 0
                task.status = "pending"
                task.result = None
                task.error = None
                task.metadata["evaluation_feedback"] = instruction
        trace.event("evaluation_repair", action=action, tasks=[task.id for task in targets])
        return {"tasks": tasks, "phase": "running", "current_task_id": ""}

    def finalize_node(state: RuntimeState) -> RuntimeState:
        pause("finalize")
        evaluation = state.get("evaluation", {})
        phase = "done" if evaluation.get("passed") else "failed"
        tasks = state.get("tasks", {})
        report = next(
            (str((task.result or {}).get("report", "")) for task in reversed(list(tasks.values())) if task.kind == "report" and task.result),
            "",
        )
        output = {
            "run_id": state["run_id"],
            "goal": state["goal"],
            "status": phase,
            "report": report,
            "evaluation": evaluation,
            "tasks": {task_id: task.to_dict() for task_id, task in tasks.items()},
            "errors": state.get("errors", []),
            "replan_count": state.get("replan_count", 0),
            "replan_history": state.get("replan_history", []),
        }
        trace.event("run_finished", phase=phase, evaluation=evaluation)
        if progress:
            progress(f"[done] phase={phase}")
        return {"phase": phase, "final_output": output}

    graph = StateGraph(RuntimeState)
    graph.add_node("plan", plan_node)
    graph.add_node("schedule", schedule_node)
    graph.add_node("execute", execute_node)
    graph.add_node("handle", handle_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("repair", repair_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "schedule")
    graph.add_conditional_edges("schedule", route_schedule, {"execute": "execute", "evaluate": "evaluate", "finalize": "finalize"})
    graph.add_edge("execute", "handle")
    graph.add_edge("handle", "schedule")
    graph.add_conditional_edges("evaluate", route_evaluate, {"repair": "repair", "finalize": "finalize"})
    graph.add_edge("repair", "schedule")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())


def run_agent(
    goal: str = "",
    *,
    mode: str = "llm",
    thread_id: str | None = None,
    checkpoint_db: str | Path | None = None,
    continue_run: bool = False,
    max_steps: int = 4,
    max_evaluations: int = 3,
    max_runtime_seconds: float = 480.0,
    evaluator_mode: str = "rules",
    memory_db: str | Path | None = None,
    trace_dir: str | Path | None = None,
    pause_before_node: str | None = None,
    pause_seconds: float = 0.0,
    progress: Progress | None = None,
) -> dict[str, Any]:
    settings = Settings.load()
    if trace_dir is not None:
        active_trace_dir = Path(trace_dir).resolve()
        active_trace_dir.mkdir(parents=True, exist_ok=True)
        settings = replace(settings, trace_dir=active_trace_dir)
    active_thread = thread_id or f"run_{uuid4().hex[:10]}"
    if continue_run and not thread_id:
        raise ValueError("--continue requires --thread-id.")
    if not continue_run and not goal.strip():
        raise ValueError("A goal is required for a new run.")
    checkpointer, connection = open_checkpointer(checkpoint_db or settings.checkpoint_db)
    trace = TraceLogger(active_thread, settings)
    client = OpenAICompatibleClient(settings)
    planner = Planner(mode, client=client)
    agent = TaskAgent(
        mode,
        client=client,
        memory=MemoryStore(memory_db or settings.memory_db),
        max_steps=max_steps,
        progress=progress,
    )
    judge_client = client if evaluator_mode == "hybrid" else None
    quality_evaluator = QualityEvaluator(judge_call=judge_client.complete_json if judge_client else None)
    graph = build_graph(
        planner,
        agent,
        quality_evaluator,
        trace,
        checkpointer=checkpointer,
        progress=progress,
        pause_before_node=pause_before_node,
        pause_seconds=pause_seconds,
        max_runtime_seconds=max_runtime_seconds,
    )
    config = {"configurable": {"thread_id": active_thread}}
    try:
        if continue_run:
            snapshot = graph.get_state(config)
            if not snapshot.values:
                raise ValueError(f"No checkpoint found for thread_id={active_thread}.")
            result = snapshot.values if not snapshot.next else graph.invoke(None, config=config)
        else:
            initial: RuntimeState = {
                "run_id": active_thread,
                "goal": goal,
                "phase": "created",
                "tasks": {},
                "current_task_id": "",
                "execution_result": {},
                "evaluation": {},
                "evaluation_count": 0,
                "max_evaluations": max(1, max_evaluations),
                "replan_count": 0,
                "replan_history": [],
                "errors": [],
                "events": [],
            }
            trace.event("run_started", goal=goal, mode=mode)
            result = graph.invoke(initial, config=config)
        output = result.get("final_output", {})
        return {
            "run_status": "finished" if output else "interrupted",
            "thread_id": active_thread,
            "phase": result.get("phase"),
            "output": output,
            "evaluation": result.get("evaluation", {}),
            "evaluation_count": result.get("evaluation_count", 0),
            "replan_count": result.get("replan_count", 0),
            "replan_history": result.get("replan_history", []),
            "tasks": {task_id: task.to_dict() for task_id, task in result.get("tasks", {}).items()},
            "trace_path": str(trace.path),
        }
    finally:
        connection.close()
