from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from durable_agent.config import Settings


def tokens(text: str) -> set[str]:
    lowered = text.lower()
    words = set(re.findall(r"[a-z0-9_+-]{2,}", lowered))
    for sequence in re.findall(r"[\u4e00-\u9fff]+", lowered):
        if len(sequence) == 1:
            words.add(sequence)
        else:
            words.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return words


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    title: str
    source: str
    text: str


class LocalRetriever:
    def __init__(self, source_dir: str | Path | None = None) -> None:
        self.source_dir = Path(source_dir) if source_dir else Settings.load().source_dir
        self._chunks: list[Chunk] | None = None

    def load(self) -> list[Chunk]:
        if self._chunks is not None:
            return self._chunks
        chunks: list[Chunk] = []
        for path in sorted(self.source_dir.glob("*.md")):
            content = path.read_text(encoding="utf-8")
            title = next((line.lstrip("# ") for line in content.splitlines() if line.startswith("#")), path.stem)
            paragraphs = [item.strip() for item in re.split(r"\n\s*\n", content) if item.strip() and not item.startswith("#")]
            for index, paragraph in enumerate(paragraphs, start=1):
                chunks.append(Chunk(f"{path.stem}#{index}", title, str(path), paragraph))
        self._chunks = chunks
        return chunks

    def search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        query_tokens = tokens(query)
        scored = []
        for chunk in self.load():
            chunk_tokens = tokens(f"{chunk.title} {chunk.text}")
            score = len(query_tokens & chunk_tokens) / max(1, len(query_tokens))
            if score:
                scored.append((score, chunk))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            {
                "citation_id": f"[{index}]",
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "url": f"local://{Path(chunk.source).name}",
                "text": chunk.text,
                "score": round(score, 4),
            }
            for index, (score, chunk) in enumerate(scored[: max(1, min(10, top_k))], start=1)
        ]

    def read(self, filename: str) -> str:
        target = (self.source_dir / filename).resolve()
        if self.source_dir.resolve() not in target.parents or target.suffix.lower() != ".md":
            raise ValueError("Only markdown files inside the source directory may be read.")
        return target.read_text(encoding="utf-8")
