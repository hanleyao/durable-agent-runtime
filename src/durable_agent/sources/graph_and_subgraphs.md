---
title: Graph Boundaries and Subgraphs
aliases: [subgraph, 子图, 子流程, graph boundary, 图边界]
domains: [LangGraph, Agent workflow]
related: [StateGraph, task graph, subagent]
confusable_with: [图论子图, graph theory subgraph, subagent]
---
# Graph Boundaries and Subgraphs

顶层 Graph 规定规划、调度、执行、处理结果、评估、修复和结束之间的合法路径。子图适合封装单个任务内部可复用的搜索、阅读和综合流程，使父图保持轻量并保留内部边界。

The top-level StateGraph owns workflow phases: plan, schedule, execute, handle, evaluate, repair, and finalize. Conditional edges enforce which transitions are legal. The model cannot jump directly from an unverified plan to a successful final state because the program owns these graph boundaries.

A subgraph is useful when one task has a reusable internal workflow with its own state and boundaries. It can hide detailed search, reading, tool-use, and synthesis steps behind one parent-graph node, reducing top-level state complexity while retaining explicit control inside the task.

In the current project, the per-task Agent Loop is a callable execution boundary rather than a separately compiled LangGraph subgraph. It still restricts research tasks to registered search tools and a finite step budget, but describing it as a fully independent persisted subgraph would overstate the implementation.
