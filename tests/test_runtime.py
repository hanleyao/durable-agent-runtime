from __future__ import annotations

import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

from durable_agent.agent import TaskAgent
from durable_agent.benchmark import load_cases, run_benchmark
from durable_agent.chat import ConversationalAgent
from durable_agent.conversation import ConversationStore
from durable_agent.config import Settings
from durable_agent.evaluator import EvaluationReport, QualityEvaluator
from durable_agent.e2e import E2ECase, load_e2e_cases, run_e2e, score_e2e_run
from durable_agent.eval_protocol import (
    create_blinded_review_pack,
    freeze_evaluation,
    initialize_heldout,
    validate_e2e_dataset,
    verify_evaluation_lock,
)
from durable_agent.fault import run_fault_trials
from durable_agent.jobs import JobStore, run_worker
from durable_agent.memory import MemoryStore
from durable_agent.models import Task
from durable_agent.planner import Planner, find_cycle, repair_plan
from durable_agent.rag import LocalRetriever
from durable_agent.runtime import build_graph, merge_replanned_tasks, run_agent
from durable_agent.trace import TraceLogger


class RuntimeTests(unittest.TestCase):
    def test_runtime_replan_replaces_failed_dag_and_records_history(self) -> None:
        class PlannerStub:
            def plan(self, goal):
                task = Task.create("broken report", "report", task_id="broken", max_attempts=1)
                return {task.id: task}, [], "initial"

            def replan(self, goal, tasks, evaluation):
                task = Task.create("corrected report", "report", task_id="corrected")
                return {task.id: task}, [], "replace failed report"

        class AgentStub:
            def execute(self, goal, task, completed):
                if task.id == "broken":
                    return {"ok": False, "error": "injected failure", "retryable": False}
                return {"ok": True, "result": {"report": "A corrected and sufficiently detailed report."}}

        class EvaluatorStub:
            def __init__(self):
                self.calls = 0

            def evaluate(self, data):
                self.calls += 1
                if self.calls == 1:
                    return EvaluationReport(False, "replan", 0.2, {}, [], [], "replace failed branch", False, None)
                return EvaluationReport(True, "pass", 1.0, {}, [], [], "", False, None)

        with tempfile.TemporaryDirectory() as directory:
            settings = replace(Settings.load(), trace_dir=Path(directory))
            trace = TraceLogger("replan-integration", settings)
            graph = build_graph(PlannerStub(), AgentStub(), EvaluatorStub(), trace)
            result = graph.invoke({
                "run_id": "replan-integration", "goal": "write report", "phase": "created",
                "tasks": {}, "current_task_id": "", "execution_result": {}, "evaluation": {},
                "evaluation_count": 0, "max_evaluations": 3, "replan_count": 0,
                "replan_history": [], "errors": [], "events": [],
            }, config={"configurable": {"thread_id": "replan-integration"}})
        self.assertEqual("done", result["phase"])
        self.assertEqual(1, result["replan_count"])
        self.assertEqual("corrected", result["replan_history"][0]["task_ids"][0])
        self.assertEqual("done", result["tasks"]["corrected"].status)

    def test_llm_replanner_builds_new_dag_and_reuses_only_unchanged_completed_work(self) -> None:
        class Client:
            def complete_json(self, system, user):
                self.payload = json.loads(user)
                return {"reasoning": "add missing evidence", "tasks": [
                    {"id": "source", "goal": "collect checkpoint evidence", "kind": "research", "blocked_by": []},
                    {"id": "extra", "goal": "collect recovery evidence", "kind": "research", "blocked_by": []},
                    {"id": "report", "goal": "write corrected report", "kind": "report", "blocked_by": ["source", "extra"]},
                ]}

        old_source = Task.create("collect checkpoint evidence", "research", task_id="source")
        old_source.status = "done"
        old_source.result = {"answer": "usable evidence"}
        old_report = Task.create("write report", "report", task_id="report", blocked_by=["source"])
        old_report.status = "failed"
        previous = {"source": old_source, "report": old_report}
        planner = Planner(client=Client())
        replanned, repairs, reasoning = planner.replan(
            "research checkpoint recovery", previous, {"action": "replan", "revision_instruction": "add evidence"},
        )
        merged, reused = merge_replanned_tasks(previous, replanned, "add evidence")
        self.assertEqual("add missing evidence", reasoning)
        self.assertEqual([], repairs)
        self.assertEqual(["source"], reused)
        self.assertEqual("done", merged["source"].status)
        self.assertEqual("pending", merged["extra"].status)
        self.assertEqual(["source", "extra"], merged["report"].blocked_by)

    def test_fault_injection_kills_and_recovers_same_thread(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_fault_trials(
                "research checkpoint recovery and generate a report",
                points=["execute"],
                mode="deterministic",
                output_dir=Path(directory) / "fault",
                window_seconds=2,
                wait_timeout=8,
            )
        trial = result["trials"][0]
        self.assertTrue(trial["fault_window_observed"])
        self.assertTrue(trial["process_killed"])
        self.assertTrue(trial["recovered"])
        self.assertEqual([], trial["duplicate_completed_tasks"])

    def test_e2e_scorer_requires_routes_topics_and_valid_citations(self) -> None:
        case = E2ECase(
            "case", "research", ["research"],
            {
                "routes": ["task"], "final_status": "done", "allowed_actions": ["pass"],
                "required_topics": ["checkpoint"], "citation_required": True, "minimum_evidence": 1,
            },
            [],
        )
        score = score_e2e_run(case, [{
            "route": "task", "task_status": "done", "answer": "Checkpoint recovery [1].",
            "evaluation": {"action": "pass", "hard_failures": []},
            "runtime_metrics": {"evidence_count": 1},
        }])
        self.assertTrue(score["passed"])

    def test_e2e_dataset_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "cases.jsonl"
            dataset.write_text(
                '{"id":"same","turns":["one"]}\n{"id":"same","turns":["two"]}\n',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_e2e_cases(dataset)

    def test_e2e_runner_isolates_state_and_writes_reproducible_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "cases.jsonl"
            dataset.write_text(
                '{"id":"direct","category":"chat","turns":["hello"],"expected":{"routes":["direct"]}}\n'
                '{"id":"task","category":"research","turns":["research checkpoint and generate report"],'
                '"expected":{"routes":["task"],"final_status":"done","allowed_actions":["pass"],'
                '"required_topics":["checkpoint"],"citation_required":true,"minimum_evidence":1}}\n',
                encoding="utf-8",
            )
            result = run_e2e(dataset, mode="deterministic", output_dir=root / "result")
            task_root = root / "result" / "artifacts" / "task" / "1"
            self.assertEqual(1.0, result["metrics"]["end_to_end_pass_rate"])
            self.assertTrue((task_root / "conversations.sqlite").exists())
            self.assertTrue((task_root / "checkpoints.sqlite").exists())
            self.assertTrue((task_root / "memory.sqlite").exists())
            self.assertTrue((root / "result" / "runs.jsonl").exists())
            self.assertEqual(["hello"], result["runs"][0]["input_turns"])

    def test_conversation_store_persists_full_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conversations.sqlite"
            first = ConversationStore(path)
            first.add_message("demo", "user", "remember checkpoint")
            first.add_message("demo", "assistant", "I will remember it.", route="direct")
            reopened = ConversationStore(path)
            history = reopened.history("demo")
        self.assertEqual(["user", "assistant"], [item.role for item in history])
        self.assertEqual("remember checkpoint", history[0].content)

    def test_conversation_delivery_source_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ConversationStore(Path(directory) / "conversations.sqlite")
            first = store.add_message("demo", "assistant", "report", source_id="job:one")
            second = store.add_message("demo", "assistant", "report", source_id="job:one")
            history = store.history("demo")
        self.assertEqual(first.id, second.id)
        self.assertEqual(1, len(history))

    def test_chat_routes_task_into_existing_runtime(self) -> None:
        class Client:
            def complete_json(self, system, user):
                self.assert_route_prompt = "Route one conversational turn" in system
                return {"route": "task", "standalone_goal": "research durable checkpoints", "reason": "research requested"}

        calls = []

        def runtime(goal, **kwargs):
            calls.append((goal, kwargs))
            return {
                "thread_id": kwargs["thread_id"],
                "output": {"report": "Checkpoint research report [1]."},
                "evaluation": {"passed": True, "action": "pass"},
            }

        with tempfile.TemporaryDirectory() as directory:
            store = ConversationStore(Path(directory) / "conversations.sqlite")
            agent = ConversationalAgent(store=store, client=Client(), runtime_call=runtime)
            result = agent.reply("请调研 checkpoint", session_id="demo")
            history = store.history("demo")
        self.assertEqual("task", result["route"])
        self.assertEqual("research durable checkpoints", calls[0][0])
        self.assertEqual("Checkpoint research report [1].", history[-1].content)
        self.assertTrue(history[-1].run_id.startswith("chat_demo_"))

    def test_chat_can_submit_task_without_blocking(self) -> None:
        submitted = []

        def submit(goal, session_id):
            submitted.append((goal, session_id))
            return {"job_id": "job_demo", "thread_id": "background_job_demo", "status": "queued"}

        with tempfile.TemporaryDirectory() as directory:
            store = ConversationStore(Path(directory) / "conversations.sqlite")
            agent = ConversationalAgent(mode="deterministic", store=store)
            result = agent.reply(
                "调研 checkpoint 并生成报告",
                session_id="demo",
                task_submit=submit,
            )
            history = store.history("demo")
        self.assertEqual([("调研 checkpoint 并生成报告", "demo")], submitted)
        self.assertEqual("job_demo", result["background_job_id"])
        self.assertEqual("queued", result["task_status"])
        self.assertIn("job_demo", history[-1].content)

    def test_deterministic_router_respects_explicit_no_research_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = ConversationalAgent(
                mode="deterministic",
                store=ConversationStore(Path(directory) / "conversations.sqlite"),
            )
            result = agent.reply("用一句话解释幂等，不需要做调研。", session_id="demo")
        self.assertEqual("direct", result["route"])

    def test_chat_compacts_context_without_deleting_audit_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ConversationStore(Path(directory) / "conversations.sqlite")
            agent = ConversationalAgent(
                mode="deterministic", store=store, recent_messages=4, summary_trigger=6,
            )
            for index in range(5):
                agent.reply(f"message {index}", session_id="compact")
            active = store.context("compact", recent_limit=4)
            history = store.history("compact", limit=20)
        self.assertTrue(active["summary"])
        self.assertLessEqual(len(active["messages"]), 4)
        self.assertEqual(10, len(history))

    def test_local_retriever_returns_citation_ready_evidence(self) -> None:
        matches = LocalRetriever().search("durable Agent checkpoint recovery", top_k=2)
        self.assertTrue(matches)
        self.assertEqual("[1]", matches[0]["citation_id"])
        self.assertTrue(matches[0]["text"])

    def test_knowledge_base_covers_core_runtime_topics(self) -> None:
        retriever = LocalRetriever()
        chunks = retriever.load()
        self.assertGreaterEqual(len({Path(chunk.source).name for chunk in chunks}), 17)
        self.assertGreaterEqual(len(chunks), 67)
        expectations = {
            "dynamic replanning completed task reuse": "dynamic_replanning#",
            "conversation session context memory": "conversation_context_memory#",
            "citation integrity unknown citations": "citation_integrity#",
            "subagent multi-agent coordination": "subagents_and_multi_agent#",
            "上下文 会话 长期记忆 检查点的区别": "conversation_context_memory#",
            "后台任务 心跳 租约 失联恢复": "background_jobs#",
        }
        for query, prefix in expectations.items():
            matches = retriever.search(query, top_k=3)
            self.assertTrue(any(item["chunk_id"].startswith(prefix) for item in matches), query)

    def test_report_rejects_unknown_citation_and_uses_global_namespace(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.calls = 0

            def complete_json(self, system, user):
                self.calls += 1
                if self.calls == 1:
                    return {"action": "final", "answer": "Invented source [3].", "used_citations": ["[3]"]}
                return {"action": "final", "answer": "Supported synthesis [1] [2].", "used_citations": ["[1]", "[2]"]}

        completed = {
            "one": {"evidence": [{"citation_id": "[1]", "chunk_id": "a#1", "text": "first"}]},
            "two": {"evidence": [{"citation_id": "[1]", "chunk_id": "b#1", "text": "second"}]},
        }
        with tempfile.TemporaryDirectory() as directory:
            client = Client()
            agent = TaskAgent(client=client, memory=MemoryStore(Path(directory) / "memory.sqlite"))
            result = agent.execute("write report", Task.create("synthesize", "report"), completed)
        self.assertTrue(result["ok"])
        self.assertEqual(2, client.calls)
        self.assertEqual(["[1]", "[2]"], [item["id"] for item in result["result"]["citations"]])
        self.assertEqual(["[1]", "[2]"], result["result"]["used_citations"])

    def test_memory_persists_and_retrieves_relevant_note(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.sqlite")
            memory_id = store.add("Checkpoint recovery resumes durable Agent workflows.")
            matches = store.search("durable checkpoint")
            self.assertEqual(memory_id, matches[0]["id"])

    def test_plan_repair_removes_invalid_dependencies_and_adds_report(self) -> None:
        tasks, repairs = repair_plan({"tasks": [
            {"id": "source", "goal": "collect", "kind": "research", "blocked_by": ["missing"]}
        ]}, "demo")
        self.assertEqual([], find_cycle(tasks))
        self.assertIn("report", tasks)
        self.assertTrue(repairs)

    def test_cycle_falls_back_to_safe_deterministic_plan(self) -> None:
        tasks, repairs = repair_plan({"tasks": [
            {"id": "a", "goal": "a", "kind": "analysis", "blocked_by": ["b"]},
            {"id": "b", "goal": "b", "kind": "analysis", "blocked_by": ["a"]},
        ]}, "demo")
        self.assertEqual(["research"], tasks["analysis"].blocked_by)
        self.assertTrue(any("cycle" in item.lower() for item in repairs))

    def test_evaluator_distinguishes_revise_from_replan(self) -> None:
        evaluator = QualityEvaluator()
        revise = evaluator.evaluate({"goal": "write report", "answer": ""})
        replan = evaluator.evaluate({
            "goal": "research checkpoint evidence",
            "answer": "Checkpoint is reliable for every workload.",
        })
        self.assertEqual("revise", revise.action)
        self.assertEqual("replan", replan.action)

    def test_llm_judge_cannot_override_deterministic_hard_failure(self) -> None:
        judge = lambda system, user: {"scores": {name: 1.0 for name in (
            "goal_coverage", "answer_quality", "groundedness", "citation_integrity",
            "execution_reliability", "calibration",
        )}}
        report = QualityEvaluator(judge_call=judge).evaluate({
            "goal": "explain checkpoint recovery",
            "answer": "Checkpoint recovery resumes persisted workflows from a durable boundary [9].",
            "citations": [{"id": "[1]"}],
            "used_citations": ["[9]"],
        })
        self.assertFalse(report.passed)
        self.assertIn("unknown_citations", report.hard_failures)

    def test_llm_judge_failure_degrades_to_rules(self) -> None:
        def broken_judge(system, user):
            raise RuntimeError("offline")

        report = QualityEvaluator(judge_call=broken_judge).evaluate({
            "goal": "explain checkpoint recovery",
            "answer": "Checkpoint recovery restores saved workflow state and resumes execution from the latest durable node boundary after interruption.",
        })
        self.assertIsNotNone(report.judge_error)
        self.assertFalse(report.judge_used)
        self.assertTrue(any(issue.code == "judge_unavailable" for issue in report.issues))

    def test_deterministic_runtime_completes_and_evaluates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "checkpoints.sqlite"
            result = run_agent(
                "explain durable Agent checkpoint recovery",
                mode="deterministic",
                checkpoint_db=database,
                thread_id="deterministic-test",
            )
            self.assertEqual("done", result["phase"])
            self.assertTrue(result["evaluation"]["passed"])
            self.assertTrue(result["output"]["report"])
            self.assertTrue(database.exists())

    def test_completed_checkpoint_can_be_reopened_without_reexecution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "checkpoints.sqlite"
            first = run_agent(
                "explain durable Agent checkpoint recovery",
                mode="deterministic",
                checkpoint_db=database,
                thread_id="resume-test",
            )
            resumed = run_agent(
                mode="deterministic",
                checkpoint_db=database,
                thread_id="resume-test",
                continue_run=True,
            )
            self.assertEqual(first["output"]["report"], resumed["output"]["report"])
            self.assertEqual("done", resumed["phase"])

    def test_job_store_idempotency_and_queued_cancellation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory) / "jobs.sqlite")
            first, created = store.create("demo", "deterministic", idempotency_key="request-1")
            second, created_again = store.create("different", "deterministic", idempotency_key="request-1")
            self.assertTrue(created)
            self.assertFalse(created_again)
            self.assertEqual(first.id, second.id)
            self.assertEqual("canceled", store.cancel(first.id).status)

    def test_background_worker_writes_result_back_to_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conversation_db = root / "conversations.sqlite"
            conversations = ConversationStore(conversation_db)
            conversations.add_message("demo", "user", "research checkpoint and generate report")
            jobs = JobStore(root / "jobs.sqlite")
            job, _ = jobs.create(
                "research checkpoint and generate report",
                "deterministic",
                session_id="demo",
                conversation_db=conversation_db,
            )
            status = run_worker(jobs.database, job.id)
            completed = jobs.get(job.id)
            history = conversations.history("demo")
            duplicate = conversations.add_message(
                "demo", "assistant", history[-1].content, source_id=f"job:{job.id}",
            )
            result_file_exists = Path(completed.result_path).exists()
        self.assertEqual("succeeded", status)
        self.assertEqual("succeeded", completed.status)
        self.assertTrue(completed.delivered_at)
        self.assertTrue(result_file_exists)
        self.assertEqual(f"job:{job.id}", history[-1].source_id)
        self.assertEqual(history[-1].id, duplicate.id)
        self.assertEqual(2, len(history))

    def test_stale_worker_lease_requeues_job(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory) / "jobs.sqlite")
            job, _ = store.create("demo", "deterministic", max_attempts=2)
            claimed = store.claim("dead-worker", 123, job.id, lease_seconds=0.01)
            self.assertEqual("running", claimed.status)
            time.sleep(0.02)
            self.assertEqual([job.id], store.recover_stale())
            self.assertEqual("queued", store.get(job.id).status)

    def test_fixed_dataset_is_versioned_and_passes_quality_gate(self) -> None:
        manifest, cases, fingerprint = load_cases()
        result = run_benchmark()
        self.assertEqual("1.0.0", manifest["version"])
        self.assertEqual(25, len(cases))
        self.assertEqual(64, len(fingerprint))
        self.assertTrue(result["gate_passed"])
        self.assertEqual(result["cases"], result["matched"])

    def test_e2e_development_dataset_is_valid_and_profiled(self) -> None:
        settings = Settings.load()
        profile = validate_e2e_dataset(settings.project_dir / "evals" / "e2e" / "dev.jsonl")
        self.assertEqual(20, profile["case_count"])
        self.assertGreaterEqual(profile["route_counts"]["task"], 12)
        self.assertGreaterEqual(profile["route_counts"]["direct"], 6)

    def test_evaluation_lock_detects_dataset_change(self) -> None:
        settings = Settings.load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "heldout.jsonl"
            dataset.write_text(
                '{"id":"one","category":"direct","turns":["hello"],'
                '"expected":{"routes":["direct"]},"human_review":["relevant"]}\n',
                encoding="utf-8",
            )
            lock_path = root / "freeze.json"
            lock = freeze_evaluation(
                dataset,
                lock_path,
                rubric=settings.project_dir / "evals" / "e2e" / "rubric.md",
                project_dir=settings.project_dir,
            )
            self.assertIn(
                "src/durable_agent/sources/checkpoint_recovery.md",
                lock["critical_code_sha256"],
            )
            self.assertTrue(verify_evaluation_lock(lock_path, project_dir=settings.project_dir)["valid"])
            dataset.write_text(dataset.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            verification = verify_evaluation_lock(lock_path, project_dir=settings.project_dir)
        self.assertFalse(verification["valid"])
        self.assertIn("dataset_sha256", verification["mismatches"])

    def test_heldout_template_and_blinded_review_pack(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = initialize_heldout(root / "heldout.jsonl", count=3)
            self.assertEqual(3, len(template.read_text(encoding="utf-8").splitlines()))
            with self.assertRaises(FileExistsError):
                initialize_heldout(template, count=3)
            runs = root / "runs.jsonl"
            runs.write_text(json.dumps({
                "case_id": "one", "repeat": 1,
                "turns": [{
                    "standalone_goal": "research checkpoints", "answer": "answer",
                    "evaluation": {"action": "pass", "overall_score": 0.99},
                }],
                "score": {"human_review": ["factually correct"]},
            }) + "\n", encoding="utf-8")
            output = root / "review.jsonl"
            create_blinded_review_pack(runs, output)
            review = json.loads(output.read_text(encoding="utf-8"))
        self.assertNotIn("evaluation", review)
        self.assertIsNone(review["accepted"])
        self.assertEqual(["factually correct"], review["review_criteria"])


if __name__ == "__main__":
    unittest.main()
