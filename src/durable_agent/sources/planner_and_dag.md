# LLM Planner and Task DAG Validation

Task System 负责决定执行什么任务以及何时执行：LLM Planner 把自然语言目标拆成带依赖的 research、analysis 和 report 任务，但计划只是提案。程序必须修复非法 ID、未知依赖、类型错误和环，再把安全 Task DAG 交给调度器执行。它与任务内部的 Subgraph 或 Agent Loop 协同，使顶层调度和内部流程保持分离。

The Planner translates a natural-language goal into typed tasks such as `research`, `analysis`, and `report`, with `blocked_by` dependencies. The LLM is a proposal generator: it chooses a useful semantic decomposition, but its output is not trusted as executable structure.

Deterministic repair validates task IDs, task kinds, dependency references, and cycles. Invalid dependencies are removed, required report work can be added, and an unsafe cyclic plan falls back to a known acyclic research-analysis-report plan. The Runtime executes only the repaired DAG.

The scheduler marks a task ready only after every dependency is done. A failed, blocked, or skipped dependency propagates blocking downstream. Task execution retries and final-answer revisions have separate budgets, preventing one failure mode from silently consuming every recovery attempt.
