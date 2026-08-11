# Evaluation-driven Agents

评估驱动的 Agent 不会因为答案非空就直接通过。质量门同时检查目标覆盖、答案质量、证据支撑、引用完整性、执行可靠性和置信度校准，并用硬失败阻止严重错误被平均分掩盖。

An Agent result should not pass merely because it is non-empty. A quality gate can combine deterministic invariants with semantic scoring. Hard failures such as empty output, unknown citations, and invalid confidence values should not be overridden by an LLM judge.

Evaluation actions have different meanings: revise rewrites an answer using existing material, replan changes the upstream task plan because evidence or execution is insufficient, and abort stops when success cannot be defined safely.
