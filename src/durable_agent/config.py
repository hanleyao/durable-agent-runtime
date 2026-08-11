from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent.parent


def load_dotenv(path: str | Path | None = None) -> None:
    candidate = Path(path) if path else PROJECT_DIR / ".env"
    if not candidate.exists():
        return
    for raw in candidate.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    project_dir: Path = PROJECT_DIR
    data_dir: Path = PROJECT_DIR / "data"
    log_dir: Path = PROJECT_DIR / "logs"
    trace_dir: Path = PROJECT_DIR / "traces"
    source_dir: Path = PACKAGE_DIR / "sources"
    checkpoint_db: Path = PROJECT_DIR / "data" / "checkpoints.sqlite"
    job_db: Path = PROJECT_DIR / "data" / "jobs.sqlite"
    memory_db: Path = PROJECT_DIR / "data" / "memory.sqlite"
    conversation_db: Path = PROJECT_DIR / "data" / "conversations.sqlite"
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()
        settings = cls(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        for directory in (settings.data_dir, settings.log_dir, settings.trace_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return settings
