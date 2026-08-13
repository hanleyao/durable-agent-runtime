from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from durable_agent.rag import LocalRetriever


@dataclass(frozen=True)
class RAGCase:
    id: str
    query: str
    relevant: tuple[str, ...]
    category: str
    language: str


def load_rag_cases(path: str | Path) -> tuple[list[RAGCase], str]:
    dataset = Path(path).resolve()
    cases: list[RAGCase] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(dataset.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {dataset}:{line_number}: {exc}") from exc
        case_id = str(item.get("id", "")).strip()
        query = str(item.get("query", "")).strip()
        relevant = tuple(str(value).strip() for value in item.get("relevant", []) if str(value).strip())
        if not case_id or case_id in seen or not query or not relevant:
            raise ValueError(f"RAG case needs a unique id, query and relevant targets: {case_id!r}")
        seen.add(case_id)
        cases.append(RAGCase(
            case_id, query, relevant, str(item.get("category", "uncategorized")),
            str(item.get("language", "unknown")),
        ))
    if not cases:
        raise ValueError("RAG dataset is empty.")
    return cases, hashlib.sha256(dataset.read_bytes()).hexdigest()


def _matches(chunk_id: str, target: str) -> bool:
    return chunk_id == target or chunk_id.startswith(target)


def validate_rag_dataset(path: str | Path, retriever: LocalRetriever | None = None) -> dict[str, Any]:
    cases, fingerprint = load_rag_cases(path)
    available = {chunk.chunk_id for chunk in (retriever or LocalRetriever()).load()}
    errors: list[str] = []
    categories: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    for case in cases:
        categories[case.category] += 1
        languages[case.language] += 1
        for target in case.relevant:
            if not any(_matches(chunk_id, target) for chunk_id in available):
                errors.append(f"{case.id}: relevant target does not match a chunk: {target}")
    if errors:
        raise ValueError("RAG dataset validation failed:\n- " + "\n- ".join(errors))
    return {
        "dataset": str(Path(path).resolve()),
        "dataset_sha256": fingerprint,
        "case_count": len(cases),
        "category_counts": dict(sorted(categories.items())),
        "language_counts": dict(sorted(languages.items())),
    }


def run_rag_evaluation(
    path: str | Path,
    *,
    output_dir: str | Path | None = None,
    top_k: int = 5,
    retriever: LocalRetriever | None = None,
) -> dict[str, Any]:
    dataset = Path(path).resolve()
    cases, fingerprint = load_rag_cases(dataset)
    active = retriever or LocalRetriever()
    limit = max(1, min(10, top_k))
    runs: list[dict[str, Any]] = []
    for case in cases:
        matches = active.search(case.query, top_k=limit)
        ranked = [str(item["chunk_id"]) for item in matches]
        relevant_ranks = [
            index for index, chunk_id in enumerate(ranked, start=1)
            if any(_matches(chunk_id, target) for target in case.relevant)
        ]
        first_rank = min(relevant_ranks) if relevant_ranks else None
        runs.append({
            "id": case.id,
            "query": case.query,
            "category": case.category,
            "language": case.language,
            "relevant": list(case.relevant),
            "retrieved": ranked,
            "first_relevant_rank": first_rank,
            "reciprocal_rank": 0.0 if first_rank is None else round(1 / first_rank, 6),
            "recall_at_1": bool(first_rank and first_rank <= 1),
            "recall_at_3": bool(first_rank and first_rank <= 3),
            "recall_at_5": bool(first_rank and first_rank <= 5),
        })

    def slice_metrics(selected: list[dict[str, Any]]) -> dict[str, float | int]:
        return {
            "cases": len(selected),
            "recall_at_1": round(mean(item["recall_at_1"] for item in selected), 6),
            "recall_at_3": round(mean(item["recall_at_3"] for item in selected), 6),
            "recall_at_5": round(mean(item["recall_at_5"] for item in selected), 6),
            "mrr": round(mean(item["reciprocal_rank"] for item in selected), 6),
        }

    metrics = slice_metrics(runs)
    by_category = {
        category: slice_metrics([item for item in runs if item["category"] == category])
        for category in sorted({item["category"] for item in runs})
    }
    by_language = {
        language: slice_metrics([item for item in runs if item["language"] == language])
        for language in sorted({item["language"] for item in runs})
    }
    result = {
        "dataset": str(dataset), "dataset_sha256": fingerprint, "top_k": limit,
        "metrics": metrics, "by_category": by_category, "by_language": by_language, "runs": runs,
    }
    if output_dir:
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        (output / "rag-metrics.json").write_text(
            json.dumps({key: value for key, value in result.items() if key != "runs"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (output / "rag-runs.jsonl").open("w", encoding="utf-8") as stream:
            for item in runs:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        result["output_dir"] = str(output)
    return result
