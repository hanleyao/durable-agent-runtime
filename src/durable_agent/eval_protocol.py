from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from durable_agent import __version__
from durable_agent.e2e import _jsonl, load_e2e_cases


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            capture_output=True, check=True, timeout=10,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _critical_files(root: Path) -> list[Path]:
    package = root / "src" / "durable_agent"
    return sorted([
        *package.glob("*.py"),
        *package.glob("sources/*.md"),
    ])


def validate_e2e_dataset(path: str | Path, *, require_complete: bool = True) -> dict[str, Any]:
    dataset = Path(path).resolve()
    cases, fingerprint = load_e2e_cases(dataset)
    errors: list[str] = []
    categories: Counter[str] = Counter()
    routes: Counter[str] = Counter()
    for case in cases:
        categories[case.category] += 1
        expected_routes = case.expected.get("routes", [])
        if not isinstance(expected_routes, list) or len(expected_routes) != len(case.turns):
            errors.append(f"{case.id}: expected.routes must contain one route per turn")
            expected_routes = []
        invalid_routes = [route for route in expected_routes if route not in {"direct", "task"}]
        if invalid_routes:
            errors.append(f"{case.id}: invalid routes {invalid_routes}")
        routes.update(str(route) for route in expected_routes)
        if "task" in expected_routes:
            if case.expected.get("final_status") != "done":
                errors.append(f"{case.id}: task case must require final_status=done")
            if not case.expected.get("allowed_actions"):
                errors.append(f"{case.id}: task case needs allowed_actions")
            if not case.expected.get("required_topics"):
                errors.append(f"{case.id}: task case needs required_topics")
            if not case.expected.get("citation_required"):
                errors.append(f"{case.id}: task case must require citations")
        if require_complete and not case.human_review:
            errors.append(f"{case.id}: human_review criteria are required")
        serialized = json.dumps({"turns": case.turns, "expected": case.expected}, ensure_ascii=False)
        if require_complete and any(token in serialized for token in ("TODO", "<填写", "<fill")):
            errors.append(f"{case.id}: contains an unfinished placeholder")
    if errors:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(errors))
    return {
        "dataset": str(dataset),
        "dataset_sha256": fingerprint,
        "case_count": len(cases),
        "category_counts": dict(sorted(categories.items())),
        "route_counts": dict(sorted(routes.items())),
    }


def freeze_evaluation(
    dataset: str | Path,
    output: str | Path,
    *,
    rubric: str | Path,
    project_dir: str | Path,
) -> dict[str, Any]:
    root = Path(project_dir).resolve()
    profile = validate_e2e_dataset(dataset)
    rubric_path = Path(rubric).resolve()
    dataset_path = Path(dataset).resolve()
    dataset_reference = str(dataset_path.relative_to(root)) if dataset_path.is_relative_to(root) else str(dataset_path)
    rubric_reference = str(rubric_path.relative_to(root)) if rubric_path.is_relative_to(root) else str(rubric_path)
    code = {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in _critical_files(root)
    }
    lock = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "agent_version": __version__,
        "git_commit": _git_commit(root),
        **{**profile, "dataset": dataset_reference},
        "rubric": rubric_reference,
        "rubric_sha256": sha256_file(rubric_path),
        "critical_code_sha256": code,
    }
    target = Path(output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return lock


def verify_evaluation_lock(lock_path: str | Path, *, project_dir: str | Path) -> dict[str, Any]:
    root = Path(project_dir).resolve()
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    mismatches: list[str] = []
    dataset = Path(lock["dataset"])
    rubric = Path(lock["rubric"])
    dataset = dataset if dataset.is_absolute() else root / dataset
    rubric = rubric if rubric.is_absolute() else root / rubric
    if not dataset.exists() or sha256_file(dataset) != lock["dataset_sha256"]:
        mismatches.append("dataset_sha256")
    if not rubric.exists() or sha256_file(rubric) != lock["rubric_sha256"]:
        mismatches.append("rubric_sha256")
    for relative, expected in lock.get("critical_code_sha256", {}).items():
        source = root / relative
        if not source.exists() or sha256_file(source) != expected:
            mismatches.append(relative)
    return {"valid": not mismatches, "mismatches": mismatches, "lock": str(Path(lock_path).resolve())}


def initialize_heldout(path: str | Path, count: int = 24) -> Path:
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing held-out file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for index in range(1, max(1, count) + 1):
            row = {
                "id": f"heldout_{index:03d}",
                "category": "<填写类别>",
                "turns": ["<填写一个真实用户请求；不要复制开发集>"],
                "expected": {
                    "routes": ["<direct或task>"],
                    "final_status": "<task填写done；direct删除此字段>",
                    "allowed_actions": ["<task通常填写pass>"],
                    "required_topics": ["<填写必须覆盖的概念>"],
                    "citation_required": "<task填写true；direct删除此字段>",
                    "minimum_evidence": "<task填写整数；direct删除此字段>",
                },
                "human_review": ["<填写人工判断答案是否可接受的语义标准>"],
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target


def create_blinded_review_pack(runs_path: str | Path, output: str | Path) -> dict[str, Any]:
    runs = _jsonl(Path(runs_path).resolve())
    target = Path(output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for run in runs:
            turns = run.get("turns", [])
            final = turns[-1] if turns else {}
            row = {
                "review_id": f"{run.get('case_id')}__r{run.get('repeat')}",
                "case_id": run.get("case_id"),
                "repeat": run.get("repeat"),
                "request": run.get("input_turns") or [turn.get("standalone_goal") for turn in turns],
                "answer": final.get("answer", ""),
                "review_criteria": run.get("score", {}).get("human_review", []),
                "accepted": None,
                "critical_issue": None,
                "reason": "",
                "reviewer_id": "",
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"runs": len(runs), "output": str(target), "predictions_hidden": True}
