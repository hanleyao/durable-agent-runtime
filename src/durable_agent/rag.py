from __future__ import annotations

import math
import re
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from durable_agent.config import Settings


def tokens(text: str) -> list[str]:
    """Tokenize English words and overlapping Chinese bigrams without external deps."""
    lowered = text.lower()
    result = re.findall(r"[a-z0-9_+-]{2,}", lowered)
    for sequence in re.findall(r"[\u4e00-\u9fff]+", lowered):
        if len(sequence) == 1:
            result.append(sequence)
        else:
            result.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return result


def vector_features(text: str) -> Counter[int]:
    """Dependency-free sparse hashing embedding over word and character n-grams."""
    lowered = re.sub(r"\s+", " ", text.lower()).strip()
    features: list[str] = [f"w:{item}" for item in tokens(lowered)]
    compact = re.sub(r"\s+", "", lowered)
    features.extend(
        f"c{size}:{compact[index:index + size]}"
        for size in (2, 3)
        for index in range(max(0, len(compact) - size + 1))
    )
    dimensions = 4096
    return Counter(
        int.from_bytes(hashlib.blake2b(item.encode("utf-8"), digest_size=8).digest(), "big") % dimensions
        for item in features
    )


def cosine(left: Counter[int], right: Counter[int]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / max(1e-9, left_norm * right_norm)


def _front_matter(content: str) -> tuple[dict[str, Any], str]:
    """Read the small YAML subset used by the bundled Markdown sources."""
    if not content.startswith("---\n"):
        return {}, content
    end = content.find("\n---\n", 4)
    if end < 0:
        return {}, content
    metadata: dict[str, Any] = {}
    current_list: str | None = None
    for raw in content[4:end].splitlines():
        if raw.startswith("  - ") and current_list:
            metadata.setdefault(current_list, []).append(raw[4:].strip().strip('"\''))
            continue
        match = re.match(r"^([a-z_]+):\s*(.*)$", raw.strip())
        if not match:
            continue
        key, value = match.groups()
        if not value:
            metadata[key] = []
            current_list = key
        elif value.startswith("[") and value.endswith("]"):
            metadata[key] = [item.strip().strip('"\'') for item in value[1:-1].split(",") if item.strip()]
            current_list = None
        else:
            metadata[key] = value.strip().strip('"\'')
            current_list = None
    return metadata, content[end + 5 :]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    title: str
    section: str
    source: str
    text: str
    aliases: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    related: tuple[str, ...] = ()
    confusable_with: tuple[str, ...] = ()


class LocalRetriever:
    """Section-aware lexical hybrid retriever for the bundled domain knowledge."""

    DEFAULT_METADATA: dict[str, dict[str, list[str]]] = {
        "graph_and_subgraphs": {
            "aliases": ["subgraph", "子图", "子流程", "graph boundary", "图边界"],
            "domains": ["LangGraph", "Agent workflow"],
            "related": ["StateGraph", "task graph", "subagent"],
            "confusable_with": ["图论子图", "graph theory subgraph", "subagent"],
        },
        "conversation_context_memory": {
            "aliases": ["memory", "context", "session", "message", "长期记忆", "上下文", "会话"],
            "domains": ["Agent memory", "conversation"],
            "related": ["context compaction", "checkpoint"],
        },
        "checkpoint_recovery": {
            "aliases": ["checkpoint", "检查点", "断点恢复", "任务持久化"],
            "domains": ["LangGraph", "durable execution"],
            "related": ["SQLite checkpointer", "thread_id", "recovery"],
        },
        "background_jobs": {
            "aliases": ["job store", "后台任务", "长任务", "heartbeat", "lease"],
            "domains": ["background jobs", "durable execution"],
            "related": ["worker", "retry", "checkpoint"],
        },
        "langchain_vs_langgraph": {
            "aliases": ["LangChain", "LangGraph", "链和图", "framework comparison", "library integrations", "stateful orchestration"],
            "domains": ["Agent framework"],
            "related": ["orchestration", "StateGraph", "agent tools"],
        },
        "idempotency_side_effects": {
            "aliases": ["idempotency", "idempotent", "幂等", "幂等性", "重复执行"],
            "domains": ["durable execution", "external side effects"],
            "related": ["checkpoint recovery", "unique constraint", "delivery record"],
        },
    }

    def __init__(self, source_dir: str | Path | None = None) -> None:
        self.source_dir = Path(source_dir) if source_dir else Settings.load().source_dir
        self._chunks: list[Chunk] | None = None
        self._document_frequency: Counter[str] = Counter()
        self._vectors: dict[str, Counter[int]] = {}

    @staticmethod
    def _values(metadata: dict[str, Any], fallback: dict[str, list[str]], key: str) -> tuple[str, ...]:
        value = metadata.get(key, fallback.get(key, []))
        if isinstance(value, str):
            value = [value]
        return tuple(str(item) for item in value or [])

    def load(self) -> list[Chunk]:
        if self._chunks is not None:
            return self._chunks
        chunks: list[Chunk] = []
        for path in sorted(self.source_dir.glob("*.md")):
            metadata, content = _front_matter(path.read_text(encoding="utf-8"))
            fallback = self.DEFAULT_METADATA.get(path.stem, {})
            title = str(metadata.get("title") or next(
                (line.lstrip("# ").strip() for line in content.splitlines() if line.startswith("# ")),
                path.stem,
            ))
            aliases = self._values(metadata, fallback, "aliases")
            domains = self._values(metadata, fallback, "domains")
            related = self._values(metadata, fallback, "related")
            confusable = self._values(metadata, fallback, "confusable_with")
            section = "Overview"
            sections: list[tuple[str, str]] = []

            for block in re.split(r"\n\s*\n", content):
                block = block.strip()
                if not block:
                    continue
                heading = re.match(r"^#{1,6}\s+(.+)$", block)
                if heading:
                    section = heading.group(1).strip()
                else:
                    # Paragraph boundaries are already curated semantic boundaries;
                    # the active heading is retained as metadata for every chunk.
                    sections.append((section, block))
            for index, (section_name, text) in enumerate(sections, start=1):
                chunks.append(Chunk(
                    f"{path.stem}#{index}", title, section_name, str(path), text,
                    aliases, domains, related, confusable,
                ))
        self._chunks = chunks
        self._document_frequency.clear()
        for chunk in chunks:
            searchable = " ".join((chunk.title, chunk.section, chunk.text, *chunk.aliases, *chunk.domains, *chunk.related))
            self._document_frequency.update(set(tokens(searchable)))
            self._vectors[chunk.chunk_id] = vector_features(searchable)
        return chunks

    def _score(self, query: str, chunk: Chunk) -> tuple[float, dict[str, float]]:
        query_tokens = tokens(query)
        if not query_tokens:
            return 0.0, {}
        corpus_size = max(1, len(self.load()))
        searchable = " ".join((chunk.title, chunk.section, chunk.text, *chunk.aliases, *chunk.domains, *chunk.related))
        counts = Counter(tokens(searchable))
        lexical = 0.0
        for token in set(query_tokens):
            if counts[token]:
                inverse_frequency = math.log(1.0 + corpus_size / (1 + self._document_frequency[token]))
                lexical += inverse_frequency * min(2, counts[token])
        lexical /= max(1.0, len(set(query_tokens)))

        lowered = query.lower()
        title_match = 1.6 if chunk.title.lower() in lowered or lowered in chunk.title.lower() else 0.0
        alias_match = max((1.8 for alias in chunk.aliases if alias.lower() in lowered), default=0.0)
        section_match = 1.1 if chunk.section.lower() in lowered or lowered in chunk.section.lower() else 0.0
        domain_match = max((0.7 for domain in chunk.domains if domain.lower() in lowered), default=0.0)
        phrase_match = 0.5 if any(
            phrase.lower() in searchable.lower()
            for phrase in re.findall(r"[a-z][a-z0-9_+-]*(?:\s+[a-z][a-z0-9_+-]*)+|[\u4e00-\u9fff]{3,}", lowered)
        ) else 0.0
        query_has_chinese = bool(re.search(r"[\u4e00-\u9fff]", query))
        text_has_chinese = bool(re.search(r"[\u4e00-\u9fff]", chunk.text))
        language_match = 1.0 if query_has_chinese == text_has_chinese else 0.0
        components = {
            "lexical": lexical,
            "title": title_match,
            "alias": alias_match,
            "section": section_match,
            "domain": domain_match,
            "phrase": phrase_match,
            "language": language_match,
        }
        return sum(components.values()), components

    def search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        self.load()
        query_vector = vector_features(query)
        candidates: list[tuple[float, float, Chunk, dict[str, float]]] = []
        for chunk in self.load():
            lexical_score, components = self._score(query, chunk)
            vector_score = cosine(query_vector, self._vectors[chunk.chunk_id])
            if lexical_score > 0 or vector_score > 0:
                candidates.append((lexical_score, vector_score, chunk, components))

        # Reciprocal-rank fusion prevents either retrieval family from dominating
        # because its raw score happens to have a larger numeric range.
        lexical_rank = {
            item[2].chunk_id: rank for rank, item in enumerate(
                sorted(candidates, key=lambda value: (-value[0], value[2].chunk_id)), start=1,
            )
        }
        vector_rank = {
            item[2].chunk_id: rank for rank, item in enumerate(
                sorted(candidates, key=lambda value: (-value[1], value[2].chunk_id)), start=1,
            )
        }
        reranked: list[tuple[float, float, float, Chunk, dict[str, float]]] = []
        lowered = query.lower()
        query_terms = set(tokens(query))
        for lexical_score, vector_score, chunk, components in candidates:
            rrf = 1 / (60 + lexical_rank[chunk.chunk_id]) + 1 / (60 + vector_rank[chunk.chunk_id])
            metadata_boost = (
                0.20 * any(alias.lower() in lowered for alias in chunk.aliases)
                + 0.08 * any(domain.lower() in lowered for domain in chunk.domains)
                + 0.04 * min(3, len(query_terms & set(tokens(chunk.title + " " + chunk.section))))
            )
            final_score = rrf * 30 + metadata_boost + vector_score * 0.35 + min(lexical_score, 4.0) * 0.08
            reranked.append((final_score, lexical_score, vector_score, chunk, components))
        reranked.sort(key=lambda item: (-item[0], item[3].chunk_id))
        return [
            {
                "citation_id": f"[{index}]",
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "section": chunk.section,
                "url": f"local://{Path(chunk.source).name}",
                "text": chunk.text,
                "aliases": list(chunk.aliases),
                "domains": list(chunk.domains),
                "confusable_with": list(chunk.confusable_with),
                "score": round(score, 4),
                "lexical_score": round(lexical_score, 4),
                "vector_score": round(vector_score, 4),
                "score_components": {key: round(value, 4) for key, value in components.items() if value},
            }
            for index, (score, lexical_score, vector_score, chunk, components) in enumerate(
                reranked[: max(1, min(10, top_k))], start=1,
            )
        ]

    @staticmethod
    def confidence(matches: list[dict[str, Any]]) -> str:
        if not matches:
            return "low"
        top = float(matches[0]["score"])
        second = float(matches[1]["score"]) if len(matches) > 1 else 0.0
        if top >= 1.15 and (top - second >= 0.08 or top >= 1.35):
            return "high"
        if top >= 0.55:
            return "medium"
        return "low"

    def read(self, filename: str) -> str:
        target = (self.source_dir / filename).resolve()
        if self.source_dir.resolve() not in target.parents or target.suffix.lower() != ".md":
            raise ValueError("Only markdown files inside the source directory may be read.")
        return target.read_text(encoding="utf-8")
