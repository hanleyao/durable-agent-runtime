from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from durable_agent.benchmark import run_benchmark
from durable_agent.chat import ConversationalAgent, format_history
from durable_agent.config import Settings
from durable_agent.conversation import ConversationStore
from durable_agent.evaluator import QualityEvaluator
from durable_agent.e2e import compare_e2e, run_e2e
from durable_agent.eval_protocol import (
    create_blinded_review_pack,
    freeze_evaluation,
    initialize_heldout,
    validate_e2e_dataset,
    verify_evaluation_lock,
)
from durable_agent.fault import DEFAULT_POINTS, load_fault_cases, run_fault_trials
from durable_agent.jobs import Job, JobStore, launch_worker, run_worker, tail
from durable_agent.memory import MemoryStore
from durable_agent.rag import LocalRetriever
from durable_agent.runtime import run_agent


def compact_job(job: Job) -> str:
    return f"{job.id}  {job.status:<16} attempt={job.attempts}/{job.max_attempts}\n  goal: {job.goal}\n  updated: {job.updated_at}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="durable-agent", description="Durable, dependency-aware and evaluation-driven Agent runtime")
    sub = parser.add_subparsers(dest="command", required=True)

    chat = sub.add_parser("chat", help="Start or continue a persistent conversation")
    chat.add_argument("message", nargs="?")
    chat.add_argument("--session", default="default")
    chat.add_argument("--mode", choices=["llm", "deterministic"], default="llm")
    chat.add_argument("--recent-messages", type=int, default=8)
    chat.add_argument("--background", action="store_true", help="Run routed task requests as durable background jobs")
    chat.add_argument("--quiet", action="store_true")
    chat.add_argument("--json", action="store_true")

    session = sub.add_parser("session", help="Inspect persistent conversation sessions")
    session_sub = session.add_subparsers(dest="session_command", required=True)
    session_list = session_sub.add_parser("list")
    session_list.add_argument("--limit", type=int, default=20)
    session_show = session_sub.add_parser("show")
    session_show.add_argument("session_id")
    session_show.add_argument("--limit", type=int, default=50)

    run = sub.add_parser("run", help="Run one goal in the foreground")
    run.add_argument("goal", nargs="?", default="")
    run.add_argument("--mode", choices=["llm", "deterministic"], default="llm")
    run.add_argument("--thread-id")
    run.add_argument("--checkpoint-db")
    run.add_argument("--continue", dest="continue_run", action="store_true")
    run.add_argument("--max-steps", type=int, default=4)
    run.add_argument("--max-evaluations", type=int, default=3)
    run.add_argument("--max-runtime-seconds", type=float, default=480.0)
    run.add_argument("--evaluator", choices=["rules", "hybrid"], default="rules")
    run.add_argument("--quiet", action="store_true")
    run.add_argument("--json", action="store_true")
    run.add_argument("--memory-db", help=argparse.SUPPRESS)
    run.add_argument("--trace-dir", help=argparse.SUPPRESS)
    run.add_argument("--pause-before-node", choices=["plan", "schedule", "execute", "handle", "evaluate", "repair", "finalize"], help=argparse.SUPPRESS)
    run.add_argument("--pause-seconds", type=float, default=0.0, help=argparse.SUPPRESS)
    run.add_argument("--output-json", help=argparse.SUPPRESS)

    resume = sub.add_parser("resume", help="Continue a persisted runtime thread")
    resume.add_argument("thread_id")
    resume.add_argument("--mode", choices=["llm", "deterministic"], default="llm")
    resume.add_argument("--checkpoint-db")
    resume.add_argument("--evaluator", choices=["rules", "hybrid"], default="rules")
    resume.add_argument("--quiet", action="store_true")
    resume.add_argument("--json", action="store_true")

    submit = sub.add_parser("submit", help="Persist a background job and return immediately")
    submit.add_argument("goal")
    submit.add_argument("--mode", choices=["llm", "deterministic"], default="llm")
    submit.add_argument("--max-attempts", type=int, default=2)
    submit.add_argument("--idempotency-key")
    submit.add_argument("--no-start", action="store_true")
    submit.add_argument("--database")
    submit.add_argument("--session")
    submit.add_argument("--conversation-db")

    for name in ("status", "cancel", "retry", "logs", "wait", "start"):
        command = sub.add_parser(name)
        command.add_argument("job_id")
        command.add_argument("--database")
        if name == "status":
            command.add_argument("--json", action="store_true")
            command.add_argument("--tail", type=int, default=0)
        if name == "logs":
            command.add_argument("--lines", type=int, default=30)
        if name == "wait":
            command.add_argument("--poll-seconds", type=float, default=1)
    listing = sub.add_parser("list")
    listing.add_argument("--database")
    listing.add_argument("--limit", type=int, default=20)
    recover = sub.add_parser("recover")
    recover.add_argument("--database")
    recover.add_argument("--start", action="store_true")
    worker = sub.add_parser("worker", help="Internal durable queue worker")
    worker.add_argument("--database")
    worker.add_argument("--job-id")
    worker.add_argument("--forever", action="store_true")

    evaluate = sub.add_parser("evaluate", help="Evaluate one saved Agent output JSON")
    evaluate.add_argument("input")
    evaluate.add_argument("--json", action="store_true")
    benchmark = sub.add_parser("benchmark", help="Run the fixed Evaluator regression set")
    benchmark.add_argument("--json", action="store_true")
    benchmark.add_argument("--output")

    e2e = sub.add_parser("e2e-run", help="Run an isolated end-to-end Agent evaluation set")
    e2e.add_argument("--dataset")
    e2e.add_argument("--mode", choices=["llm", "deterministic"], default="deterministic")
    e2e.add_argument("--variant", choices=["full", "single_pass", "no_quality_loop"], default="full")
    e2e.add_argument("--repeats", type=int, default=1)
    e2e.add_argument("--case", action="append", dest="cases")
    e2e.add_argument("--limit", type=int)
    e2e.add_argument("--output")
    e2e.add_argument("--min-pass-rate", type=float, default=1.0)
    e2e.add_argument("--quiet", action="store_true")
    e2e.add_argument("--json", action="store_true")
    e2e.add_argument("--lock", help="Require a valid frozen evaluation lock before running")

    comparison = sub.add_parser("e2e-compare", help="Compare single-pass, no-quality-loop and full Agent variants")
    comparison.add_argument("--dataset")
    comparison.add_argument("--mode", choices=["llm", "deterministic"], default="deterministic")
    comparison.add_argument("--repeats", type=int, default=1)
    comparison.add_argument("--case", action="append", dest="cases")
    comparison.add_argument("--limit", type=int)
    comparison.add_argument("--output")
    comparison.add_argument("--quiet", action="store_true")
    comparison.add_argument("--json", action="store_true")

    fault = sub.add_parser("fault-test", help="Kill Agent processes at checkpointed boundaries and verify recovery")
    fault.add_argument("--goal", default="research durable Agent checkpoint recovery and generate a report")
    fault.add_argument("--dataset")
    fault.add_argument("--point", action="append", choices=["schedule", "execute", "handle", "evaluate", "repair", "finalize"])
    fault.add_argument("--repeats", type=int, default=1)
    fault.add_argument("--mode", choices=["llm", "deterministic"], default="deterministic")
    fault.add_argument("--output")
    fault.add_argument("--window-seconds", type=float, default=20.0)
    fault.add_argument("--wait-timeout", type=float, default=30.0)
    fault.add_argument("--quiet", action="store_true")
    fault.add_argument("--json", action="store_true")

    dataset_validate = sub.add_parser("dataset-validate", help="Validate and profile an end-to-end dataset")
    dataset_validate.add_argument("--dataset")
    dataset_validate.add_argument("--json", action="store_true")
    evaluation_freeze = sub.add_parser("eval-freeze", help="Freeze dataset, rubric and critical Agent code fingerprints")
    evaluation_freeze.add_argument("--dataset")
    evaluation_freeze.add_argument("--rubric")
    evaluation_freeze.add_argument("--output", required=True)
    evaluation_verify = sub.add_parser("eval-verify", help="Verify a frozen evaluation lock")
    evaluation_verify.add_argument("lock")
    heldout = sub.add_parser("heldout-init", help="Create a private held-out authoring worksheet")
    heldout.add_argument("--output")
    heldout.add_argument("--count", type=int, default=24)
    review = sub.add_parser("review-pack", help="Create a blinded human-review file from E2E runs")
    review.add_argument("runs")
    review.add_argument("--output", required=True)

    memory = sub.add_parser("memory", help="Add or search long-term memory")
    memory_sub = memory.add_subparsers(dest="memory_command", required=True)
    add = memory_sub.add_parser("add")
    add.add_argument("content")
    add.add_argument("--kind", default="note")
    search = memory_sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=5)

    knowledge = sub.add_parser("knowledge", help="Inspect or search the local evidence knowledge base")
    knowledge_sub = knowledge.add_subparsers(dest="knowledge_command", required=True)
    knowledge_sub.add_parser("list")
    knowledge_search = knowledge_sub.add_parser("search")
    knowledge_search.add_argument("query")
    knowledge_search.add_argument("--limit", type=int, default=5)
    knowledge_search.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings = Settings.load()
    if args.command == "chat":
        store = ConversationStore(settings.conversation_db)
        agent = ConversationalAgent(
            mode=args.mode,
            store=store,
            recent_messages=args.recent_messages,
        )
        progress = None if args.quiet else lambda message: print(message, flush=True)
        job_store = JobStore(settings.job_db) if args.background else None

        def submit_task(goal: str, session_id: str) -> dict[str, object]:
            assert job_store is not None
            job, _ = job_store.create(
                goal, args.mode, session_id=session_id, conversation_db=store.path,
            )
            worker_pid = launch_worker(job_store.database, job.id)
            return {"job_id": job.id, "thread_id": job.thread_id, "status": job.status, "worker_pid": worker_pid}

        def respond(message: str) -> int:
            try:
                result = agent.reply(
                    message,
                    session_id=args.session,
                    progress=progress,
                    task_submit=submit_task if args.background else None,
                )
            except Exception as exc:
                print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 2
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(f"\nagent> {result['answer']}")
                suffix = f" run={result['run_id']}" if result.get("run_id") else ""
                if result.get("background_job_id"):
                    suffix += f" job={result['background_job_id']}"
                print(f"\n[session={result['session_id']} route={result['route']}{suffix}]")
            return 0

        if args.message is not None:
            return respond(args.message)
        if args.json:
            print("error: --json requires a one-shot message.", file=sys.stderr)
            return 2
        print(f"DURABLE AGENT CHAT  session={args.session}")
        print("Commands: /history, /exit")
        while True:
            try:
                message = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not message:
                continue
            if message.lower() in {"/exit", "/quit"}:
                return 0
            if message.lower() == "/history":
                print(format_history(store.history(args.session, 50)) or "no messages")
                continue
            respond(message)

    if args.command == "session":
        store = ConversationStore(settings.conversation_db)
        if args.session_command == "list":
            sessions = store.list_sessions(args.limit)
            if not sessions:
                print("no sessions")
            for item in sessions:
                print(f"{item['id']}  messages={item['message_count']}  updated={item['updated_at']}  {item['title']}")
            return 0
        messages = store.history(args.session_id, args.limit)
        print(format_history(messages) or "session not found or empty")
        return 0

    if args.command in {"run", "resume"}:
        progress = None if args.quiet else lambda message: print(message, flush=True)
        is_resume = args.command == "resume"
        try:
            result = run_agent(
                "" if is_resume else args.goal,
                mode=args.mode,
                thread_id=args.thread_id,
                checkpoint_db=args.checkpoint_db,
                continue_run=True if is_resume else args.continue_run,
                max_steps=4 if is_resume else args.max_steps,
                max_evaluations=3 if is_resume else args.max_evaluations,
                max_runtime_seconds=480.0 if is_resume else args.max_runtime_seconds,
                evaluator_mode=args.evaluator,
                memory_db=None if is_resume else args.memory_db,
                trace_dir=None if is_resume else args.trace_dir,
                pause_before_node=None if is_resume else args.pause_before_node,
                pause_seconds=0.0 if is_resume else args.pause_seconds,
                progress=progress,
            )
        except Exception as exc:
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        output_json = getattr(args, "output_json", None)
        if output_json:
            output_path = Path(output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            evaluation = result.get("evaluation", {})
            print(f"\nDURABLE AGENT\nrun={result['thread_id']} status={result['phase']}")
            if evaluation:
                print(f"evaluation={evaluation.get('action')} score={evaluation.get('overall_score', 0):.2f}")
            print(f"report: {result.get('output', {}).get('report', '')}")
            print(f"trace: {result['trace_path']}")
        return 0 if result.get("phase") == "done" else 1

    if args.command in {"submit", "status", "list", "cancel", "retry", "logs", "wait", "start", "recover", "worker"}:
        database = Path(getattr(args, "database", None) or settings.job_db).resolve()
        if args.command == "worker":
            status = run_worker(database, args.job_id, args.forever)
            print(status)
            return 0 if status in {"idle", "succeeded", "canceled"} else 1
        store = JobStore(database)
        if args.command == "submit":
            job, created = store.create(
                args.goal, args.mode, args.max_attempts, args.idempotency_key,
                session_id=args.session, conversation_db=args.conversation_db,
            )
            print(f"{'submitted' if created else 'existing'} {job.id} status={job.status}")
            if not args.no_start and job.status == "queued":
                print(f"worker_pid={launch_worker(database, job.id)}")
            return 0
        if args.command == "list":
            print("\n".join(compact_job(job) for job in store.list(args.limit)) or "no jobs")
            return 0
        if args.command == "recover":
            recovered = store.recover_stale()
            print("recovered: " + (", ".join(recovered) if recovered else "none"))
            if recovered and args.start:
                print(f"worker_pid={launch_worker(database)}")
            return 0
        job = store.get(args.job_id)
        if not job:
            print("job not found", file=sys.stderr)
            return 2
        if args.command == "status":
            print(json.dumps(job.to_dict(), ensure_ascii=False, indent=2) if args.json else compact_job(job))
            if args.tail:
                print(tail(job.log_path, args.tail))
            return 0 if job.status != "failed" else 1
        if args.command == "start":
            if job.status != "queued":
                print(f"job is {job.status}; only queued jobs can start", file=sys.stderr)
                return 1
            print(f"worker_pid={launch_worker(database, job.id)}")
            return 0
        if args.command == "wait":
            last = None
            while True:
                current = store.get(job.id)
                if current is None:
                    return 2
                marker = (current.status, current.attempts)
                if marker != last:
                    print(compact_job(current), flush=True)
                    last = marker
                if current.status in {"succeeded", "failed", "canceled"}:
                    return 0 if current.status in {"succeeded", "canceled"} else 1
                time.sleep(max(0.2, args.poll_seconds))
        if args.command == "cancel":
            print(compact_job(store.cancel(job.id)))  # type: ignore[arg-type]
            return 0
        if args.command == "retry":
            retried = store.retry(job.id)
            print(compact_job(retried))  # type: ignore[arg-type]
            if retried and retried.status == "queued":
                print(f"worker_pid={launch_worker(database, retried.id)}")
            return 0
        print(tail(job.log_path, args.lines) or "log is empty")
        return 0

    if args.command == "evaluate":
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        report = QualityEvaluator().evaluate(payload)
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"evaluation={'PASS' if report.passed else 'FAIL'} action={report.action} score={report.overall_score:.2f}")
            for issue in report.issues:
                print(f"  [{issue.severity}] {issue.code}: {issue.message}")
        return 0 if report.passed else 1

    if args.command == "benchmark":
        result = run_benchmark()
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            for item in result["results"]:
                print(f"[{'OK' if item['matched'] else 'MISS'}] {item['id']:<32} expected={item['expected']:<7} actual={item['actual']:<7}")
            print(f"cases={result['cases']} action_accuracy={result['metrics']['action_accuracy']:.1%} macro_f1={result['metrics']['macro_f1']:.3f}")
            print(f"quality_gate={'PASS' if result['gate_passed'] else 'FAIL'}")
        return 0 if result["gate_passed"] else 1

    if args.command == "e2e-run":
        dataset = Path(args.dataset or settings.project_dir / "evals" / "e2e" / "dev.jsonl")
        progress = None if args.quiet else lambda message: print(message, flush=True)
        try:
            if args.lock:
                verification = verify_evaluation_lock(args.lock, project_dir=settings.project_dir)
                if not verification["valid"]:
                    raise ValueError(f"Evaluation lock mismatch: {verification['mismatches']}")
            result = run_e2e(
                dataset,
                mode=args.mode,
                variant=args.variant,
                repeats=max(1, args.repeats),
                output_dir=args.output,
                case_ids=set(args.cases or []),
                limit=args.limit,
                progress=progress,
            )
        except Exception as exc:
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        metrics = result["metrics"]
        if args.json:
            print(json.dumps({"manifest": result["manifest"], "metrics": metrics, "output_dir": result["output_dir"]}, ensure_ascii=False, indent=2))
        else:
            print("\nE2E EVALUATION")
            print(f"cases={metrics['cases']} runs={metrics['runs']} mode={result['manifest']['mode']} variant={result['manifest']['variant']}")
            print(f"end_to_end_pass_rate={metrics['end_to_end_pass_rate']:.1%}")
            print(f"task_completion_rate={metrics['task_completion_rate']:.1%}")
            print(f"route_accuracy={metrics['route_accuracy']:.1%}")
            print(f"first_pass_rate={metrics['first_pass_rate']:.1%}")
            print(f"citation_validity_rate={metrics['citation_validity_rate']:.1%}")
            print(f"stable_case_rate={metrics['stable_case_rate']:.1%}")
            print(f"output={result['output_dir']}")
        return 0 if metrics["end_to_end_pass_rate"] >= max(0.0, min(1.0, args.min_pass_rate)) else 1

    if args.command == "e2e-compare":
        dataset = Path(args.dataset or settings.project_dir / "evals" / "e2e" / "dev.jsonl")
        progress = None if args.quiet else lambda message: print(message, flush=True)
        try:
            result = compare_e2e(
                dataset,
                mode=args.mode,
                repeats=max(1, args.repeats),
                output_dir=args.output,
                case_ids=set(args.cases or []),
                limit=args.limit,
                progress=progress,
            )
        except Exception as exc:
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("\nAGENT VARIANT COMPARISON")
            for variant, metrics in result["metrics"].items():
                print(
                    f"{variant:<16} e2e={metrics['end_to_end_pass_rate']:.1%} "
                    f"task={metrics['task_completion_rate']:.1%} "
                    f"citation={metrics['citation_validity_rate']:.1%} "
                    f"duration={metrics['average_duration_seconds']:.2f}s"
                )
            print(f"output={result['output_dir']}")
        return 0

    if args.command == "fault-test":
        progress = None if args.quiet else lambda message: print(message, flush=True)
        try:
            workloads = load_fault_cases(args.dataset) if args.dataset else None
            result = run_fault_trials(
                args.goal,
                cases=workloads,
                points=args.point or list(DEFAULT_POINTS),
                repeats=max(1, args.repeats),
                mode=args.mode,
                output_dir=args.output,
                window_seconds=max(1.0, args.window_seconds),
                wait_timeout=max(1.0, args.wait_timeout),
                progress=progress,
            )
        except Exception as exc:
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            metrics = result["metrics"]
            print("\nFAULT INJECTION")
            print(f"trials={metrics['trials']} recovered={metrics['recovered']}")
            print(f"recovery_success_rate={metrics['recovery_success_rate']:.1%}")
            print(f"duplicate_execution_rate={metrics['duplicate_execution_rate']:.1%}")
            print(f"average_recovery_seconds={metrics['average_recovery_seconds']}")
            print(f"output={result['output_dir']}")
        return 0 if result["metrics"]["recovery_success_rate"] == 1.0 else 1

    if args.command == "dataset-validate":
        dataset = Path(args.dataset or settings.project_dir / "evals" / "e2e" / "dev.jsonl")
        try:
            profile = validate_e2e_dataset(dataset)
        except Exception as exc:
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(profile, ensure_ascii=False, indent=2))
        else:
            print(f"valid cases={profile['case_count']} sha256={profile['dataset_sha256']}")
            print(f"categories={json.dumps(profile['category_counts'], ensure_ascii=False)}")
            print(f"routes={json.dumps(profile['route_counts'], ensure_ascii=False)}")
        return 0

    if args.command == "eval-freeze":
        dataset = Path(args.dataset or settings.project_dir / "evals" / "e2e" / "dev.jsonl")
        rubric = Path(args.rubric or settings.project_dir / "evals" / "e2e" / "rubric.md")
        try:
            lock = freeze_evaluation(dataset, args.output, rubric=rubric, project_dir=settings.project_dir)
        except Exception as exc:
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(lock, ensure_ascii=False, indent=2))
        return 0

    if args.command == "eval-verify":
        result = verify_evaluation_lock(args.lock, project_dir=settings.project_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["valid"] else 1

    if args.command == "heldout-init":
        output = Path(args.output or settings.project_dir / "evals" / "private" / "heldout-v1.jsonl")
        try:
            target = initialize_heldout(output, args.count)
        except Exception as exc:
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        print(f"created {target}")
        print("Fill this file yourself, then run dataset-validate. Do not commit it.")
        return 0

    if args.command == "review-pack":
        try:
            result = create_blinded_review_pack(args.runs, args.output)
        except Exception as exc:
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "knowledge":
        retriever = LocalRetriever()
        if args.knowledge_command == "list":
            grouped: dict[str, dict[str, object]] = {}
            for chunk in retriever.load():
                filename = Path(chunk.source).name
                item = grouped.setdefault(filename, {"title": chunk.title, "chunks": 0})
                item["chunks"] = int(item["chunks"]) + 1
            for filename, item in grouped.items():
                print(f"{filename:<36} chunks={item['chunks']:<2}  {item['title']}")
            print(f"files={len(grouped)} chunks={sum(int(item['chunks']) for item in grouped.values())}")
            return 0
        matches = retriever.search(args.query, args.limit)
        if args.json:
            print(json.dumps(matches, ensure_ascii=False, indent=2))
        else:
            for item in matches:
                print(f"{item['citation_id']} {item['chunk_id']} score={item['score']:.4f} {item['title']}")
                print(f"    {item['text']}")
        return 0

    memory_store = MemoryStore()
    if args.memory_command == "add":
        print(f"memory_id={memory_store.add(args.content, args.kind)}")
    else:
        print(json.dumps(memory_store.search(args.query, args.limit), ensure_ascii=False, indent=2))
    return 0
