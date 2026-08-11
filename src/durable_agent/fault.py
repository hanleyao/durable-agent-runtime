from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable
from uuid import uuid4

from durable_agent import __version__
from durable_agent.config import Settings
from durable_agent.runtime import run_agent


Progress = Callable[[str], None]
DEFAULT_POINTS = ("execute", "handle", "evaluate", "finalize")


def _git_commit(project_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_dir, capture_output=True,
            text=True, check=True, timeout=10,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def load_fault_cases(path: str | Path) -> list[dict[str, str]]:
    cases = []
    seen: set[str] = set()
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid fault JSONL at line {line_number}: {exc}") from exc
        case_id = str(item.get("id", "")).strip()
        goal = str(item.get("goal", "")).strip()
        if not case_id or case_id in seen or not goal:
            raise ValueError(f"Fault cases require unique id and goal at line {line_number}.")
        seen.add(case_id)
        cases.append({"id": case_id, "goal": goal})
    return cases


def _events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(raw)
            if isinstance(item, dict):
                events.append(item)
        except json.JSONDecodeError:
            continue
    return events


def _wait_for_window(path: Path, node: str, process: subprocess.Popen[Any], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(item.get("type") == "fault_window" and item.get("node") == node for item in _events(path)):
            return True
        if process.poll() is not None:
            return False
        time.sleep(0.05)
    return False


def _duplicate_completed(events: list[dict[str, Any]]) -> list[str]:
    window = next((index for index, item in enumerate(events) if item.get("type") == "fault_window"), None)
    if window is None:
        return []
    completed = {
        str(item.get("task_id")) for item in events[:window]
        if item.get("type") == "task_result" and item.get("outcome") == "done"
    }
    redispatched = {
        str(item.get("task_id")) for item in events[window + 1:] if item.get("type") == "task_dispatched"
    }
    return sorted(completed & redispatched)


def run_fault_trials(
    goal: str | None = None,
    *,
    cases: list[dict[str, str]] | None = None,
    points: list[str] | None = None,
    repeats: int = 1,
    mode: str = "deterministic",
    output_dir: str | Path | None = None,
    window_seconds: float = 20.0,
    wait_timeout: float = 30.0,
    progress: Progress | None = None,
) -> dict[str, Any]:
    settings = Settings.load()
    workloads = cases or [{"id": "custom", "goal": str(goal or "").strip()}]
    if not workloads or any(not item.get("id") or not item.get("goal") for item in workloads):
        raise ValueError("At least one fault workload with id and goal is required.")
    selected = points or list(DEFAULT_POINTS)
    invalid = sorted(set(selected) - {"schedule", "execute", "handle", "evaluate", "repair", "finalize"})
    if invalid:
        raise ValueError(f"Unsupported fault points: {invalid}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(output_dir or settings.project_dir / "results" / "fault" / stamp).resolve()
    output.mkdir(parents=True, exist_ok=True)
    trials: list[dict[str, Any]] = []
    combinations = (
        (workload, point, repeat)
        for workload in workloads
        for point in selected
        for repeat in range(1, max(1, repeats) + 1)
    )
    for workload, point, repeat in combinations:
            case_id = workload["id"]
            active_goal = workload["goal"]
            trial_id = f"{case_id}_{point}_{repeat}_{uuid4().hex[:6]}"
            thread_id = f"fault_{trial_id}"
            root = output / "artifacts" / case_id / point / str(repeat)
            trace_dir = root / "traces"
            root.mkdir(parents=True, exist_ok=True)
            trace_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_db = root / "checkpoints.sqlite"
            memory_db = root / "memory.sqlite"
            trace_path = trace_dir / f"{thread_id}.jsonl"
            log_path = root / "interrupted.log"
            command = [
                sys.executable, "-m", "durable_agent", "run", active_goal,
                "--mode", mode, "--thread-id", thread_id,
                "--checkpoint-db", str(checkpoint_db), "--memory-db", str(memory_db),
                "--trace-dir", str(trace_dir), "--pause-before-node", point,
                "--pause-seconds", str(window_seconds), "--quiet",
            ]
            if progress:
                progress(f"[fault] case={case_id} point={point} repeat={repeat}/{max(1, repeats)}")
            observed = False
            killed = False
            error = None
            recovered_result: dict[str, Any] = {}
            recovery_started = None
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command, cwd=settings.project_dir, stdout=log,
                    stderr=subprocess.STDOUT, text=True,
                )
                try:
                    observed = _wait_for_window(trace_path, point, process, wait_timeout)
                    if observed and process.poll() is None:
                        process.kill()
                        process.wait(timeout=10)
                        killed = True
                    elif not observed:
                        error = "Fault window was not observed before timeout or process exit."
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=10)
            if observed and killed:
                try:
                    recovery_started = time.perf_counter()
                    recovered_result = run_agent(
                        mode=mode, thread_id=thread_id, checkpoint_db=checkpoint_db,
                        memory_db=memory_db, trace_dir=trace_dir, continue_run=True,
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            recovery_seconds = round(time.perf_counter() - recovery_started, 4) if recovery_started else None
            events = _events(trace_path)
            duplicates = _duplicate_completed(events)
            recovered = bool(
                observed and killed and checkpoint_db.exists() and not error and recovered_result.get("phase") == "done"
                and recovered_result.get("evaluation", {}).get("passed") and not duplicates
            )
            trials.append({
                "id": trial_id, "case_id": case_id, "goal": active_goal,
                "point": point, "repeat": repeat, "thread_id": thread_id,
                "fault_window_observed": observed, "process_killed": killed,
                "checkpoint_exists": checkpoint_db.exists(), "recovered": recovered,
                "final_phase": recovered_result.get("phase"),
                "evaluation_action": recovered_result.get("evaluation", {}).get("action"),
                "duplicate_completed_tasks": duplicates, "recovery_seconds": recovery_seconds,
                "error": error, "trace_path": str(trace_path),
            })
            if progress:
                progress(f"[fault-result] {point} -> {'recovered' if recovered else 'failed'}")
    recovery_times = [item["recovery_seconds"] for item in trials if item["recovery_seconds"] is not None]
    metrics = {
        "trials": len(trials),
        "recovered": sum(item["recovered"] for item in trials),
        "recovery_success_rate": sum(item["recovered"] for item in trials) / max(1, len(trials)),
        "duplicate_execution_rate": sum(bool(item["duplicate_completed_tasks"]) for item in trials) / max(1, len(trials)),
        "fault_window_rate": sum(item["fault_window_observed"] for item in trials) / max(1, len(trials)),
        "average_recovery_seconds": round(mean(recovery_times), 4) if recovery_times else None,
    }
    result = {
        "manifest": {
            "schema_version": "1.0.0", "agent_version": __version__,
            "agent_commit": _git_commit(settings.project_dir),
            "workload_sha256": hashlib.sha256(
                json.dumps(workloads, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "workloads": workloads, "mode": mode, "points": selected, "repeats": max(1, repeats),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "metrics": metrics, "trials": trials, "output_dir": str(output),
    }
    (output / "fault-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Fault Injection Report", "", f"- Trials: {metrics['trials']}",
        f"- Recovery success rate: {metrics['recovery_success_rate']:.1%}",
        f"- Duplicate execution rate: {metrics['duplicate_execution_rate']:.1%}",
        f"- Average recovery time: {metrics['average_recovery_seconds']}s", "",
        "| Case | Point | Repeat | Window | Killed | Recovered | Duplicates |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for item in trials:
        lines.append(
            f"| {item['case_id']} | {item['point']} | {item['repeat']} | {item['fault_window_observed']} | "
            f"{item['process_killed']} | {item['recovered']} | {', '.join(item['duplicate_completed_tasks']) or '-'} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
