from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from durable_agent.evaluator import QualityEvaluator


DATASET_DIR = Path(__file__).resolve().parent / "eval_set"
ACTIONS = ("pass", "revise", "replan", "abort")


def load_cases(dataset_dir: str | Path = DATASET_DIR) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    root = Path(dataset_dir)
    manifest_path = root / "manifest.json"
    case_path = root / "cases.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = [json.loads(line) for line in case_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [case.get("id") for case in cases]
    if len(ids) != len(set(ids)) or any(not item for item in ids):
        raise ValueError("Evaluation case IDs must be non-empty and unique.")
    labels = {case.get("expected", {}).get("action") for case in cases}
    if labels != set(ACTIONS):
        raise ValueError(f"Dataset must cover all actions; found {sorted(labels)}.")
    fingerprint = hashlib.sha256(manifest_path.read_bytes() + case_path.read_bytes()).hexdigest()
    return manifest, cases, fingerprint


def run_benchmark(dataset_dir: str | Path = DATASET_DIR) -> dict[str, Any]:
    manifest, cases, fingerprint = load_cases(dataset_dir)
    evaluator = QualityEvaluator()
    results = []
    confusion = {expected: {actual: 0 for actual in ACTIONS} for expected in ACTIONS}
    issue_total = issue_hits = 0
    for case in cases:
        report = evaluator.evaluate(case["input"])
        expected = case["expected"]
        expected_issues = set(expected.get("required_issue_codes", []))
        actual_issues = {issue.code for issue in report.issues}
        issue_total += len(expected_issues)
        issue_hits += len(expected_issues & actual_issues)
        confusion[expected["action"]][report.action] += 1
        matched = report.action == expected["action"] and report.passed == expected["passed"] and expected_issues <= actual_issues
        results.append({
            "id": case["id"], "expected": expected["action"], "actual": report.action,
            "matched": matched, "score": report.overall_score,
            "missing_issues": sorted(expected_issues - actual_issues),
        })
    per_action = {}
    for action in ACTIONS:
        true_positive = confusion[action][action]
        predicted = sum(confusion[label][action] for label in ACTIONS)
        support = sum(confusion[action].values())
        precision = true_positive / predicted if predicted else 0
        recall = true_positive / support if support else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        per_action[action] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "support": support}
    matched = sum(item["matched"] for item in results)
    action_accuracy = sum(item["expected"] == item["actual"] for item in results) / len(results)
    macro_f1 = sum(item["f1"] for item in per_action.values()) / len(ACTIONS)
    issue_recall = issue_hits / issue_total if issue_total else 1.0
    minimum = manifest.get("minimum_metrics", {})
    metrics = {"action_accuracy": round(action_accuracy, 4), "macro_f1": round(macro_f1, 4), "required_issue_recall": round(issue_recall, 4)}
    failures = [f"{name} below {value}" for name, value in minimum.items() if metrics.get(name, 0) < value]
    return {
        "dataset": {"name": manifest["name"], "version": manifest["version"], "sha256": fingerprint},
        "cases": len(cases), "matched": matched,
        "distribution": dict(Counter(case["expected"]["action"] for case in cases)),
        "metrics": {**metrics, "per_action": per_action},
        "confusion_matrix": confusion,
        "gate_passed": not failures,
        "gate_failures": failures,
        "results": results,
    }
