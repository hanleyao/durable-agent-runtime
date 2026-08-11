# Durable Agent Runtime

耐久 Agent 将语义决策与确定性执行控制分开：模型负责理解目标和提出计划，Runtime 负责依赖校验、工具限制、状态持久化、重试预算和合法状态转换。

A durable Agent separates semantic decisions from deterministic execution controls. The model may propose a task plan, while the runtime validates dependencies, limits tools, persists state, and controls retries.

Checkpoint persistence stores workflow state at node boundaries. A reopened process can load the same thread and continue unfinished work. Checkpoints do not automatically make external side effects idempotent.

Background job persistence is different from workflow checkpoint persistence. The job store tracks queue ownership, heartbeat, lease, cancellation, and attempts; the checkpoint store tracks business execution progress.
