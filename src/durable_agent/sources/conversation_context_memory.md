---
title: Conversation, Session, Context, and Memory
aliases: [memory, context, session, message, 长期记忆, 上下文, 会话]
domains: [Agent memory, conversation]
related: [context compaction, checkpoint]
confusable_with: [workflow checkpoint, message history]
---
# Conversation, Session, Context, and Memory

会话 Session 是跨轮次的持久对话边界，上下文 Context 是某次模型调用实际看到的摘要和近期消息，长期记忆 Memory 是跨会话复用的精选知识，检查点 Checkpoint 则保存一次 Graph 执行进度；四者生命周期和用途不同。

A conversation `session_id` groups durable user and assistant messages across turns. A session may create many background Jobs and many LangGraph threads. It is a user-facing continuity boundary, not a workflow execution identifier.

Context is the selected information sent to the model for one decision. This project builds active context from a rolling summary plus recent messages. Context is temporary and bounded even though the complete conversation history remains in SQLite for audit.

Long-term Memory is a separate SQLite store for reusable notes that may be retrieved across sessions and workflow runs. Message history records what was said, Memory records selected reusable knowledge, and a checkpoint records graph execution state; none of these stores is interchangeable.
