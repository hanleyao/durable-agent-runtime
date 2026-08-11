# Evaluation-driven Agents

An Agent result should not pass merely because it is non-empty. A quality gate can combine deterministic invariants with semantic scoring. Hard failures such as empty output, unknown citations, and invalid confidence values should not be overridden by an LLM judge.

Evaluation actions have different meanings: revise rewrites an answer using existing material, replan changes the upstream task plan because evidence or execution is insufficient, and abort stops when success cannot be defined safely.
