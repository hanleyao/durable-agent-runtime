# Dynamic Replanning

动态重规划在证据、执行或上游任务结构不足时生成替代 DAG。新计划仍需确定性校验；只有任务 ID、类型、目标和依赖全部未变化的已完成任务才能安全复用。

`revise` and `replan` solve different problems. Revise keeps the existing evidence and plan, then resets report-producing work so the final answer can be rewritten. Replan is used when upstream evidence, execution, or task structure is insufficient.

During replanning, the Planner receives the original goal, previous tasks and results, failures, and Evaluator feedback. The replacement DAG passes through the same deterministic validation and repair rules as the initial plan before the Runtime accepts it.

Completed work is reused only when task ID, kind, normalized goal, and dependency list are unchanged. Changed tasks receive evaluation feedback and execute again. The Runtime records `replan_count`, generation history, repairs, replacement task IDs, and reused task IDs for audit.
