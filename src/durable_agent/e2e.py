from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from durable_agent import __version__
from durable_agent.baseline import run_single_pass
from durable_agent.chat import ConversationalAgent
from durable_agent.config import Settings
from durable_agent.conversation import ConversationStore
from durable_agent.runtime import run_agent


Progress = Callable[[str], None]


@dataclass(frozen=True)
class E2ECase:
    id: str
    category: str
    turns: list[str]
    expected: dict[str, Any]
    human_review: list[str]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"Case at {path}:{line_number} must be an object.")
        rows.append(item)
    return rows


def load_e2e_cases(path: str | Path) -> tuple[list[E2ECase], str]:
    dataset = Path(path).resolve()
    raw = dataset.read_bytes()
    cases = []
    seen: set[str] = set()
    for item in _jsonl(dataset):
        case_id = str(item.get("id", "")).strip()
        turns = [str(turn).strip() for turn in item.get("turns", []) if str(turn).strip()]
        if not case_id or case_id in seen or not turns:
            raise ValueError(f"Every case needs a unique id and at least one turn: {case_id!r}")
        seen.add(case_id)
        cases.append(E2ECase(
            case_id,
            str(item.get("category", "uncategorized")),
            turns,
            item.get("expected", {}) if isinstance(item.get("expected"), dict) else {},
            [str(value) for value in item.get("human_review", [])],
        ))
    return cases, hashlib.sha256(raw).hexdigest()


def _topic_passes(answer: str, specification: Any) -> tuple[bool, str]:
    if isinstance(specification, str):
        name, alternatives = specification, [specification]
    elif isinstance(specification, dict):
        name = str(specification.get("name", "topic"))
        alternatives = [str(value) for value in specification.get("any_of", [])]
    else:
        return False, "invalid topic specification"
    lowered = answer.lower()
    passed = any(value.lower() in lowered for value in alternatives if value)
    return passed, name


def score_e2e_run(case: E2ECase, turns: list[dict[str, Any]], error: str | None = None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, actual: Any, expected: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "actual": actual, "expected": expected})

    if error:
        check("execution_error", False, error, None)
        return {"passed": False, "checks": checks, "human_review": case.human_review}
    expected_routes = [str(value) for value in case.expected.get("routes", [])]
    actual_routes = [str(turn.get("route")) for turn in turns]
    if expected_routes:
        check("routes", actual_routes == expected_routes, actual_routes, expected_routes)
    final = turns[-1] if turns else {}
    answer = str(final.get("answer", ""))
    expected_status = case.expected.get("final_status")
    if expected_status is not None:
        check("final_status", final.get("task_status") == expected_status, final.get("task_status"), expected_status)
    allowed_actions = [str(value) for value in case.expected.get("allowed_actions", [])]
    if allowed_actions:
        actual_action = final.get("evaluation", {}).get("action")
        check("evaluation_action", actual_action in allowed_actions, actual_action, allowed_actions)
    minimum_score = case.expected.get("minimum_evaluation_score")
    if minimum_score is not None:
        actual_score = float(final.get("evaluation", {}).get("overall_score", 0) or 0)
        check("evaluation_score", actual_score >= float(minimum_score), actual_score, minimum_score)
    for topic in case.expected.get("required_topics", []):
        passed, name = _topic_passes(answer, topic)
        check(f"topic:{name}", passed, name if passed else "missing", name)
    if case.expected.get("citation_required"):
        markers = sorted(set(re.findall(r"\[\d+\]", answer)))
        hard = set(final.get("evaluation", {}).get("hard_failures", []))
        check("citation_present", bool(markers), markers, "at least one citation")
        check("citation_valid", "unknown_citations" not in hard, sorted(hard), "no unknown citations")
    minimum_evidence = case.expected.get("minimum_evidence")
    if minimum_evidence is not None:
        actual_evidence = int(final.get("runtime_metrics", {}).get("evidence_count", 0) or 0)
        check("evidence_count", actual_evidence >= int(minimum_evidence), actual_evidence, minimum_evidence)
    check("answer_nonempty", bool(answer.strip()), len(answer.strip()), "> 0")
    return {
        "passed": bool(checks) and all(item["passed"] for item in checks),
        "checks": checks,
        "human_review": case.human_review,
    }


def _git_commit(project_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_dir, text=True,
            capture_output=True, check=True, timeout=10,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_report(path: Path, result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    lines = [
        "# End-to-End Evaluation Report",
        "",
        f"- Dataset: `{result['manifest']['dataset']}`",
        f"- Dataset SHA256: `{result['manifest']['dataset_sha256']}`",
        f"- Agent commit: `{result['manifest'].get('agent_commit') or 'unknown'}`",
        f"- Mode: `{result['manifest']['mode']}`",
        f"- Variant: `{result['manifest']['variant']}`",
        f"- Cases / runs: {metrics['cases']} / {metrics['runs']}",
        f"- End-to-end pass rate: {metrics['end_to_end_pass_rate']:.1%}",
        f"- Task completion rate: {metrics['task_completion_rate']:.1%}",
        f"- Route accuracy: {metrics['route_accuracy']:.1%}",
        f"- First-pass rate: {metrics['first_pass_rate']:.1%}",
        f"- Citation validity rate: {metrics['citation_validity_rate']:.1%}",
        f"- Stable case rate: {metrics['stable_case_rate']:.1%}",
        f"- Average duration: {metrics['average_duration_seconds']:.2f}s",
        "",
        "## Cases",
        "",
        "| Case | Repeat | Result | Duration |",
        "|---|---:|---:|---:|",
    ]
    for run in result["runs"]:
        lines.append(
            f"| {run['case_id']} | {run['repeat']} | {'PASS' if run['score']['passed'] else 'FAIL'} | {run['duration_seconds']:.2f}s |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_e2e(cases: list[E2ECase], runs: list[dict[str, Any]]) -> dict[str, Any]:
    route_checks = [
        item for run in runs for item in run["score"]["checks"] if item["name"] == "routes"
    ]
    task_runs = [run for run in runs if "task" in run["actual_routes"]]
    completed_task_runs = [
        run for run in task_runs
        if run["turns"] and run["turns"][-1].get("task_status") == "done" and run["score"]["passed"]
    ]
    first_pass = [
        run for run in completed_task_runs
        if int(run["turns"][-1].get("runtime_metrics", {}).get("evaluation_count", 0) or 0) <= 1
    ]
    citation_checks = [
        item for run in runs for item in run["score"]["checks"] if item["name"] == "citation_valid"
    ]
    stable = 0
    for case in cases:
        selected = [run for run in runs if run["case_id"] == case.id]
        stable += bool(selected) and all(run["score"]["passed"] for run in selected)
    return {
        "cases": len(cases),
        "runs": len(runs),
        "end_to_end_pass_rate": sum(run["score"]["passed"] for run in runs) / max(1, len(runs)),
        "task_completion_rate": len(completed_task_runs) / max(1, len(task_runs)),
        "route_accuracy": sum(item["passed"] for item in route_checks) / max(1, len(route_checks)),
        "first_pass_rate": len(first_pass) / max(1, len(completed_task_runs)),
        "citation_validity_rate": sum(item["passed"] for item in citation_checks) / max(1, len(citation_checks)),
        "stable_case_rate": stable / max(1, len(cases)),
        "average_duration_seconds": round(mean(run["duration_seconds"] for run in runs), 4) if runs else 0.0,
    }


def run_e2e(
    dataset: str | Path,
    *,
    mode: str = "deterministic",
    variant: str = "full",
    repeats: int = 1,
    output_dir: str | Path | None = None,
    case_ids: set[str] | None = None,
    limit: int | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    settings = Settings.load()
    if variant not in {"full", "single_pass", "no_quality_loop"}:
        raise ValueError(f"Unsupported E2E variant: {variant}")
    dataset_path = Path(dataset).resolve()
    cases, fingerprint = load_e2e_cases(dataset_path)
    if case_ids:
        cases = [case for case in cases if case.id in case_ids]
    if limit is not None:
        cases = cases[: max(0, limit)]
    if not cases:
        raise ValueError("No E2E cases selected.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(output_dir or settings.project_dir / "results" / "e2e" / stamp).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for case in cases:
        for repeat in range(1, max(1, repeats) + 1):
            if progress:
                progress(f"[e2e] {case.id} repeat={repeat}/{max(1, repeats)}")
            run_root = output / "artifacts" / case.id / str(repeat)
            run_root.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            turn_results: list[dict[str, Any]] = []
            error = None
            try:
                store = ConversationStore(run_root / "conversations.sqlite")

                def isolated_runtime(goal: str, **kwargs: Any) -> dict[str, Any]:
                    if variant == "single_pass":
                        return run_single_pass(goal, **kwargs)
                    return run_agent(
                        goal,
                        **kwargs,
                        checkpoint_db=run_root / "checkpoints.sqlite",
                        memory_db=run_root / "memory.sqlite",
                        trace_dir=run_root / "traces",
                        max_evaluations=1 if variant == "no_quality_loop" else 3,
                    )

                agent = ConversationalAgent(mode=mode, store=store, runtime_call=isolated_runtime)
                session_id = f"eval_{case.id}_{repeat}"
                for message in case.turns:
                    turn_results.append(agent.reply(message, session_id=session_id))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            duration = round(time.perf_counter() - started, 4)
            score = score_e2e_run(case, turn_results, error)
            runs.append({
                "case_id": case.id,
                "category": case.category,
                "repeat": repeat,
                "input_turns": case.turns,
                "duration_seconds": duration,
                "actual_routes": [turn.get("route") for turn in turn_results],
                "turns": turn_results,
                "error": error,
                "score": score,
            })
            if progress:
                progress(f"[e2e-result] {case.id} -> {'pass' if score['passed'] else 'fail'} ({duration:.2f}s)")
    manifest = {
        "schema_version": "1.0.0",
        "agent_version": __version__,
        "agent_commit": _git_commit(settings.project_dir),
        "model": settings.model if mode == "llm" else "deterministic",
        "model_base_url": settings.base_url if mode == "llm" else None,
        "dataset": str(dataset_path),
        "dataset_sha256": fingerprint,
        "mode": mode,
        "variant": variant,
        "repeats": max(1, repeats),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    result = {"manifest": manifest, "metrics": summarize_e2e(cases, runs), "runs": runs}
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "metrics.json", result["metrics"])
    with (output / "runs.jsonl").open("w", encoding="utf-8") as stream:
        for run in runs:
            stream.write(json.dumps(run, ensure_ascii=False, default=str) + "\n")
    _write_report(output / "report.md", result)
    result["output_dir"] = str(output)
    return result


def compare_e2e(
    dataset: str | Path,
    *,
    mode: str = "deterministic",
    repeats: int = 1,
    output_dir: str | Path | None = None,
    case_ids: set[str] | None = None,
    limit: int | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    settings = Settings.load()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(output_dir or settings.project_dir / "results" / "comparison" / stamp).resolve()
    variants = ("single_pass", "no_quality_loop", "full")
    results = {
        variant: run_e2e(
            dataset,
            mode=mode,
            variant=variant,
            repeats=repeats,
            output_dir=output / variant,
            case_ids=case_ids,
            limit=limit,
            progress=progress,
        )
        for variant in variants
    }
    metrics = {variant: result["metrics"] for variant, result in results.items()}
    baseline = metrics["single_pass"]
    comparable = (
        "end_to_end_pass_rate", "task_completion_rate", "route_accuracy",
        "first_pass_rate", "citation_validity_rate", "stable_case_rate",
    )
    deltas = {
        variant: {name: round(values[name] - baseline[name], 6) for name in comparable}
        for variant, values in metrics.items()
        if variant != "single_pass"
    }
    comparison = {
        "dataset": str(Path(dataset).resolve()),
        "mode": mode,
        "repeats": max(1, repeats),
        "metrics": metrics,
        "delta_vs_single_pass": deltas,
        "output_dir": str(output),
    }
    _write_json(output / "comparison.json", comparison)
    lines = [
        "# Agent Variant Comparison",
        "",
        "| Variant | E2E pass | Task completion | First pass | Citation validity | Avg duration |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in variants:
        item = metrics[variant]
        lines.append(
            f"| {variant} | {item['end_to_end_pass_rate']:.1%} | {item['task_completion_rate']:.1%} | "
            f"{item['first_pass_rate']:.1%} | {item['citation_validity_rate']:.1%} | {item['average_duration_seconds']:.2f}s |"
        )
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return comparison
