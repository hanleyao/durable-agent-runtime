# Quality Evaluator and Decision Policy

Evaluator 用六个维度和确定性硬失败共同决定结果。`pass` 接受答案，`revise` 使用现有材料重写，`replan` 修复上游任务或证据问题，`abort` 在无法合理定义成功时停止。

The Evaluator scores goal coverage, answer quality, groundedness, citation integrity, execution reliability, and calibration. A weighted overall score is necessary but not sufficient: every required dimension must clear its threshold and no deterministic hard failure may be present.

Hard failures include an empty goal, empty answer, program fallback, unknown citation, and confidence outside the range zero to one. Research without evidence lowers groundedness, failed task branches lower execution reliability, and undisclosed missing evidence lowers calibration.

The action policy maps failure causes to control flow. `pass` accepts the report, `revise` rewrites with existing work, `replan` changes upstream tasks when evidence or execution is inadequate, and `abort` stops when a valid success condition is unavailable. Evaluation and repair counts are bounded.
