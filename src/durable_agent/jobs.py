from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from durable_agent.config import Settings
from durable_agent.conversation import ConversationStore
from durable_agent.models import now_iso


TERMINAL = {"succeeded", "failed", "canceled"}


@dataclass
class Job:
    id: str
    goal: str
    mode: str
    status: str
    attempts: int
    max_attempts: int
    thread_id: str
    checkpoint_db: str
    log_path: str
    created_at: str
    updated_at: str
    worker_id: str | None
    worker_pid: int | None
    heartbeat_at: float | None
    lease_expires_at: float | None
    cancel_requested: bool
    error: str | None
    idempotency_key: str | None
    session_id: str | None
    conversation_db: str | None
    result_path: str | None
    delivered_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobStore:
    def __init__(self, database: str | Path | None = None) -> None:
        self.settings = Settings.load()
        self._custom_database = (
            database is not None
            and Path(database).resolve() != self.settings.job_db.resolve()
        )
        self.database = Path(database or self.settings.job_db).resolve()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(
                """CREATE TABLE IF NOT EXISTS jobs(
                    id TEXT PRIMARY KEY, goal TEXT NOT NULL, mode TEXT NOT NULL,
                    status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 2, thread_id TEXT NOT NULL,
                    checkpoint_db TEXT NOT NULL, log_path TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    worker_id TEXT, worker_pid INTEGER, heartbeat_at REAL,
                    lease_expires_at REAL, cancel_requested INTEGER NOT NULL DEFAULT 0,
                    error TEXT, idempotency_key TEXT UNIQUE, session_id TEXT,
                    conversation_db TEXT, result_path TEXT, delivered_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at);
                CREATE TABLE IF NOT EXISTS job_events(
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL, timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL, data_json TEXT NOT NULL DEFAULT '{}'
                );"""
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(jobs)")}
            for name in ("session_id", "conversation_db", "result_path", "delivered_at"):
                if name not in columns:
                    connection.execute(f"ALTER TABLE jobs ADD COLUMN {name} TEXT")

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.database, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=10000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _job(row: sqlite3.Row | None) -> Job | None:
        if row is None:
            return None
        return Job(
            id=row["id"], goal=row["goal"], mode=row["mode"], status=row["status"],
            attempts=row["attempts"], max_attempts=row["max_attempts"], thread_id=row["thread_id"],
            checkpoint_db=row["checkpoint_db"], log_path=row["log_path"], created_at=row["created_at"],
            updated_at=row["updated_at"], worker_id=row["worker_id"], worker_pid=row["worker_pid"],
            heartbeat_at=row["heartbeat_at"], lease_expires_at=row["lease_expires_at"],
            cancel_requested=bool(row["cancel_requested"]), error=row["error"],
            idempotency_key=row["idempotency_key"],
            session_id=row["session_id"], conversation_db=row["conversation_db"],
            result_path=row["result_path"], delivered_at=row["delivered_at"],
        )

    @staticmethod
    def _event(connection: sqlite3.Connection, job_id: str, event_type: str, **data: Any) -> None:
        connection.execute(
            "INSERT INTO job_events(job_id,timestamp,event_type,data_json) VALUES(?,?,?,?)",
            (job_id, now_iso(), event_type, json.dumps(data, ensure_ascii=False, default=str)),
        )

    def create(
        self,
        goal: str,
        mode: str = "llm",
        max_attempts: int = 2,
        idempotency_key: str | None = None,
        session_id: str | None = None,
        conversation_db: str | Path | None = None,
    ) -> tuple[Job, bool]:
        if not goal.strip():
            raise ValueError("Goal is empty.")
        with self.connect() as connection:
            if idempotency_key:
                existing = connection.execute("SELECT * FROM jobs WHERE idempotency_key=?", (idempotency_key,)).fetchone()
                if existing:
                    return self._job(existing), False  # type: ignore[return-value]
            job_id = f"job_{uuid4().hex[:12]}"
            created = now_iso()
            thread_id = f"background_{job_id}"
            if self._custom_database:
                checkpoint = str((self.database.parent / "checkpoints.sqlite").resolve())
                log_path = str((self.database.parent / "logs" / f"{job_id}.log").resolve())
                result_path = str((self.database.parent / "job-results" / f"{job_id}.json").resolve())
            else:
                checkpoint = str(self.settings.checkpoint_db.resolve())
                log_path = str((self.settings.log_dir / f"{job_id}.log").resolve())
                result_path = str((self.settings.data_dir / "job-results" / f"{job_id}.json").resolve())
            active_conversation_db = str(Path(conversation_db or self.settings.conversation_db).resolve()) if session_id else None
            connection.execute(
                """INSERT INTO jobs(id,goal,mode,status,attempts,max_attempts,thread_id,checkpoint_db,
                log_path,created_at,updated_at,idempotency_key,session_id,conversation_db,result_path)
                VALUES(?,?,?,'queued',0,?,?,?,?,?,?,?,?,?,?)""",
                (
                    job_id, goal, mode, max(1, max_attempts), thread_id, checkpoint,
                    log_path, created, created, idempotency_key, session_id,
                    active_conversation_db, result_path,
                ),
            )
            self._event(connection, job_id, "submitted", mode=mode)
            return self._job(connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()), True  # type: ignore[return-value]

    def get(self, job_id: str) -> Job | None:
        with self.connect() as connection:
            return self._job(connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def list(self, limit: int = 20) -> list[Job]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (min(200, max(1, limit)),)).fetchall()
            return [self._job(row) for row in rows]  # type: ignore[misc]

    def recover_stale(self) -> list[str]:
        recovered = []
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT * FROM jobs WHERE status IN ('running','cancel_requested') AND lease_expires_at < ?",
                (time.time(),),
            ).fetchall()
            for row in rows:
                if row["cancel_requested"]:
                    status, error = "canceled", "Lease expired after cancellation request."
                elif row["attempts"] < row["max_attempts"]:
                    status, error = "queued", "Recovered stale worker lease."
                    recovered.append(row["id"])
                else:
                    status, error = "failed", "Worker lease expired and retry budget was exhausted."
                connection.execute(
                    "UPDATE jobs SET status=?,updated_at=?,worker_id=NULL,worker_pid=NULL,heartbeat_at=NULL,lease_expires_at=NULL,error=? WHERE id=?",
                    (status, now_iso(), error, row["id"]),
                )
                self._event(connection, row["id"], "lease_recovered", status=status)
        return recovered

    def claim(self, worker_id: str, pid: int, job_id: str | None = None, lease_seconds: float = 10) -> Job | None:
        self.recover_stale()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM jobs WHERE status='queued'" + (" AND id=?" if job_id else " ORDER BY created_at LIMIT 1"),
                (job_id,) if job_id else (),
            ).fetchone()
            if not row:
                return None
            now = time.time()
            updated = connection.execute(
                """UPDATE jobs SET status='running',attempts=attempts+1,updated_at=?,worker_id=?,worker_pid=?,
                heartbeat_at=?,lease_expires_at=?,cancel_requested=0,error=NULL WHERE id=? AND status='queued'""",
                (now_iso(), worker_id, pid, now, now + lease_seconds, row["id"]),
            )
            if updated.rowcount != 1:
                return None
            self._event(connection, row["id"], "claimed", worker_id=worker_id, pid=pid)
            return self._job(connection.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone())

    def heartbeat(self, job_id: str, worker_id: str, lease_seconds: float = 10) -> bool:
        now = time.time()
        with self.connect() as connection:
            updated = connection.execute(
                "UPDATE jobs SET heartbeat_at=?,lease_expires_at=?,updated_at=? WHERE id=? AND worker_id=? AND status IN ('running','cancel_requested')",
                (now, now + lease_seconds, now_iso(), job_id, worker_id),
            )
            return updated.rowcount == 1

    def cancel(self, job_id: str) -> Job | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return None
            if row["status"] == "queued":
                connection.execute("UPDATE jobs SET status='canceled',cancel_requested=1,updated_at=?,error='Canceled before execution.' WHERE id=?", (now_iso(), job_id))
            elif row["status"] == "running":
                connection.execute("UPDATE jobs SET status='cancel_requested',cancel_requested=1,updated_at=? WHERE id=?", (now_iso(), job_id))
            self._event(connection, job_id, "cancel_requested")
            return self._job(connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def finish(self, job_id: str, worker_id: str, status: str, error: str | None = None) -> bool:
        with self.connect() as connection:
            updated = connection.execute(
                """UPDATE jobs SET status=?,error=?,updated_at=?,worker_id=NULL,worker_pid=NULL,
                heartbeat_at=NULL,lease_expires_at=NULL WHERE id=? AND worker_id=?""",
                (status, error, now_iso(), job_id, worker_id),
            )
            if updated.rowcount:
                self._event(connection, job_id, status, error=error)
            return updated.rowcount == 1

    def mark_delivered(self, job_id: str) -> None:
        with self.connect() as connection:
            connection.execute("UPDATE jobs SET delivered_at=?,updated_at=? WHERE id=?", (now_iso(), now_iso(), job_id))
            self._event(connection, job_id, "conversation_delivered")

    def fail_or_retry(self, job: Job, worker_id: str, error: str) -> str:
        current = self.get(job.id)
        status = "canceled" if current and current.cancel_requested else ("queued" if job.attempts < job.max_attempts else "failed")
        self.finish(job.id, worker_id, status, error)
        return status

    def retry(self, job_id: str) -> Job | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return None
            if row["status"] not in TERMINAL:
                return self._job(row)
            thread_id = f"background_{job_id}_retry_{uuid4().hex[:8]}"
            connection.execute(
                "UPDATE jobs SET status='queued',attempts=0,thread_id=?,cancel_requested=0,error=NULL,updated_at=? WHERE id=?",
                (thread_id, now_iso(), job_id),
            )
            self._event(connection, job_id, "manual_retry")
            return self._job(connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())


def stop_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def run_worker(database: str | Path | None = None, job_id: str | None = None, forever: bool = False) -> str:
    store = JobStore(database)
    worker_id = f"worker_{uuid4().hex[:10]}"
    while True:
        job = store.claim(worker_id, os.getpid(), job_id=job_id)
        if not job:
            if forever:
                time.sleep(2)
                continue
            return "idle"
        command = [
            sys.executable, "-m", "durable_agent", "run", job.goal,
            "--mode", job.mode, "--thread-id", job.thread_id,
            "--checkpoint-db", job.checkpoint_db, "--quiet",
            "--output-json", str(job.result_path),
        ]
        if job.attempts > 1:
            command = [
                sys.executable, "-m", "durable_agent", "run",
                "--mode", job.mode, "--thread-id", job.thread_id,
                "--checkpoint-db", job.checkpoint_db, "--continue", "--quiet",
                "--output-json", str(job.result_path),
            ]
        log_path = Path(job.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{now_iso()}] attempt={job.attempts}/{job.max_attempts}\n")
            log.flush()
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            while process.poll() is None:
                current = store.get(job.id)
                if current is None or current.cancel_requested:
                    stop_process_tree(process)
                    store.finish(job.id, worker_id, "canceled", "Canceled by operator.")
                    return "canceled"
                if not store.heartbeat(job.id, worker_id):
                    stop_process_tree(process)
                    return "lost_lease"
                time.sleep(2)
            if process.returncode == 0:
                try:
                    if job.session_id and job.conversation_db:
                        payload = json.loads(Path(str(job.result_path)).read_text(encoding="utf-8"))
                        answer = str(payload.get("output", {}).get("report", "")).strip()
                        if not answer:
                            raise RuntimeError("Completed background task has no report to deliver.")
                        ConversationStore(job.conversation_db).add_message(
                            job.session_id, "assistant", answer, route="task",
                            run_id=job.thread_id, source_id=f"job:{job.id}",
                        )
                        store.mark_delivered(job.id)
                    store.finish(job.id, worker_id, "succeeded")
                    status = "succeeded"
                except Exception as exc:
                    status = store.fail_or_retry(job, worker_id, f"Conversation delivery failed: {type(exc).__name__}: {exc}")
            else:
                status = store.fail_or_retry(job, worker_id, f"Runtime exited with code {process.returncode}.")
        if job_id and status == "queued":
            time.sleep(1)
            continue
        if not forever or status not in {"succeeded", "failed", "canceled"}:
            return status


def launch_worker(database: str | Path, job_id: str | None = None) -> int:
    command = [sys.executable, "-m", "durable_agent", "worker", "--database", str(database)]
    if job_id:
        command.extend(["--job-id", job_id])
    else:
        command.append("--forever")
    kwargs: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(command, **kwargs).pid


def tail(path: str, lines: int = 30) -> str:
    target = Path(path)
    if not target.exists() or lines <= 0:
        return ""
    with target.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        position = stream.tell()
        chunks: list[bytes] = []
        newline_count = 0
        while position > 0 and newline_count <= lines:
            size = min(8192, position)
            position -= size
            stream.seek(position)
            chunk = stream.read(size)
            chunks.append(chunk)
            newline_count += chunk.count(b"\n")
    return "\n".join(b"".join(reversed(chunks)).decode("utf-8", errors="replace").splitlines()[-lines:])
