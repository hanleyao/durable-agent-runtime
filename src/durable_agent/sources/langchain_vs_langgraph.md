---
title: LangChain and LangGraph
aliases: [LangChain, LangGraph, 链和图, framework comparison]
domains: [Agent framework]
related: [orchestration, StateGraph, agent tools]
confusable_with: [competing replacements]
---
# LangChain and LangGraph

LangChain 和 LangGraph 处于不同抽象层。LangChain 提供较高层的 Agent 框架、预构建 Agent 架构，以及统一的模型、工具和集成接口，适合快速构建常见的 LLM 与工具调用应用。LangGraph 是较低层的 Agent 编排框架和 Runtime，适合需要显式状态、节点与边、持久化、长任务恢复、流式处理和 human-in-the-loop 的复杂工作流。

LangChain Agent 构建在 LangGraph 之上，从而获得持久执行、流式输出、人工介入和状态持久化等能力。反过来，LangGraph 可以独立使用，也可以在节点中复用 LangChain 的模型和工具组件，因此两者通常是高层组件与低层编排 Runtime 的协作关系，而不是只能二选一。

选择取决于控制需求。标准 Agent Loop、模型切换、工具接入和快速开发通常优先考虑 LangChain；需要自定义 Graph 状态转换、确定性流程与 Agent 决策混合、长时间运行和故障恢复时更适合直接使用 LangGraph。简单应用没有必要为了使用 Graph 而增加编排复杂度，复杂有状态流程也不应只依赖高层默认抽象。

Official references: [LangChain overview](https://docs.langchain.com/oss/python/langchain/overview), [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview), and [Frameworks, runtimes, and harnesses](https://docs.langchain.com/oss/python/concepts/products).
