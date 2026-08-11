from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from durable_agent.agent import TaskAgent
from durable_agent.benchmark import load_cases, run_benchmark
from durable_agent.chat import ConversationalAgent
from durable_agent.conversation import ConversationStore
from durable_agent.evaluator import QualityEvaluator
from durable_agent.jobs import JobStore
from durable_agent.memory import MemoryStore
from durable_agent.models import Task
from durable_agent.planner import find_cycle, repair_plan
from durable_agent.rag import LocalRetriever
from durable_agent.runtime import run_agent


class RuntimeTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
