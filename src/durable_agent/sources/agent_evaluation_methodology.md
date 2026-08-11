# Agent Evaluation Methodology

公开开发集可以参与调试，不能作为未见测试指标。正式 held-out 必须在运行前完成任务和标签，并与评分规则、知识库和关键代码一起冻结；看到结果后修改系统会使该测试集失去未见属性。

A visible development set supports prompt, rule, and implementation debugging, so its score is not a held-out generalization metric. A held-out set must be authored and labeled before final execution, then sealed together with the rubric, evidence sources, and critical Agent code fingerprints.

End-to-end task completion should require more than a terminal process state. Checks can include correct routing, workflow success, accepted Evaluator action, required topic coverage, minimum evidence, citation validity, and a non-empty answer. Human acceptance remains separate because string checks cannot establish semantic correctness.

Useful comparisons include a single-pass baseline, a complete task runtime without a quality loop, and the full Agent. Final reporting should include sample size, repeated runs, task completion, human acceptance, first-pass rate, citation validity, recovery success, duplicate execution, latency, and uncertainty intervals.
