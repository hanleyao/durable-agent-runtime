---
title: Checkpoint and Recovery
aliases: [checkpoint, 检查点, 断点恢复, 任务持久化]
domains: [LangGraph, durable execution]
related: [SQLite checkpointer, thread_id, recovery]
confusable_with: [job store, process liveness]
---
# Checkpoint and Recovery

工作流检查点在可持久化的节点边界保存 Graph 状态，并通过 `thread_id` 定位一次执行。进程重启后可以读取已完成任务、未完成任务和评估状态，从最近的可靠进度继续运行。

A workflow checkpoint is a persisted snapshot of graph state at a durable node boundary. In this runtime the state includes the goal, Task DAG, task statuses and results, evaluation state, counters, errors, and replan history. LangGraph stores snapshots in SQLite and selects the workflow by `thread_id`.

Recovery reopens the same checkpoint database and invokes the same `thread_id` with continuation enabled. The scheduler reads persisted task states, so tasks already marked `done` are not normally dispatched again and unfinished work continues from the latest durable state.

A checkpoint represents workflow progress, not operating-system process liveness. It does not prove that a Worker is alive, does not replace a background queue, and cannot by itself make an external API call or file write exactly once.
