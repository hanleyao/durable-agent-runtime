# Evidence and Citation Integrity

检索证据拥有稳定 chunk ID，并在当前任务中分配 `[1]` 等引用编号。多分支证据汇总时需要去重和统一重编号；答案出现未知编号属于硬失败，但编号合法仍不等于证据在语义上充分支持结论。

Local retrieval returns evidence chunks with stable chunk IDs and run-local citation labels such as `[1]`. When evidence from multiple research branches is merged, it is deduplicated and assigned one citation namespace so two unrelated chunks do not both remain `[1]`.

The Agent may use only citation IDs present in available evidence. Inline markers and declared `used_citations` are checked against known citations. An unknown marker is rejected inside the Agent Loop and is also a deterministic hard failure in the final Evaluator.

Citation validity is different from factual completeness. A marker can be structurally valid while the evidence is weak or irrelevant, so human review must still judge whether claims are actually supported. Conversely, evidence that is collected but never cited produces a quality warning.
