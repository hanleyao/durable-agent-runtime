# Context Compaction

上下文压缩保留目标、约束、决定、未解决问题和关键任务结果，丢弃问候、重复与冗长日志。压缩后的摘要供模型使用，但完整原始消息仍留在 SQLite 中用于审计和评测。

Context compaction reduces model input without deleting the audit history. After the unsummarized message count crosses a threshold, older messages are summarized while the most recent messages remain verbatim. `summary_through` records the last message incorporated into the summary.

A useful summary preserves user goals, constraints, names, decisions, unresolved questions, and important task outcomes. Greetings, repetition, verbose tool logs, and wording that no longer affects future decisions can be dropped.

Compaction is lossy by design, so the full transcript stays in the Conversation Store. The system can inspect original messages for debugging and evaluation even though the model normally receives only the compact active context.
