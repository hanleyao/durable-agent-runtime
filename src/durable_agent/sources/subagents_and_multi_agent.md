# Subagents and Multi-Agent Systems

Subagent 具有独立任务、上下文、工具和结果契约；Multi-Agent 还需要处理所有权、通信、共享记忆冲突、重复劳动、死锁和归因。当前项目只有受控 Task DAG，没有实现独立 Subagent 进程或通用多 Agent 通信协议。

A subagent is an independently scoped worker with its own task, context, tools, and result contract. It can reduce context interference and allow parallel work, but the parent still needs delegation rules, budgets, result validation, cancellation, and failure propagation.

A multi-agent system adds coordination problems beyond one Planner and one Task Runtime: ownership, message protocols, shared-memory conflicts, duplicate work, deadlocks, consensus, and attribution. More agents do not automatically improve quality; they add latency and failure surfaces that require evaluation.

The current Durable Agent Runtime does not implement independent subagent processes or a general multi-agent communication protocol. Its Task DAG may contain multiple branches, but those tasks execute through one controlled runtime. Multi-agent claims should therefore be presented as future work, not an existing capability.
