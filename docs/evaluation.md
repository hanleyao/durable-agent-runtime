# Evaluation Workflow

## Dataset roles

- `eval_set/`: visible synthetic regression cases used while developing rules.
- private held-out set: unseen human labels used once for generalization metrics.
- end-to-end task set: user goals that run through Planner, Agent, tools, Runtime and Evaluator.

Never report the visible regression-set score as real-world accuracy.

## Recommended held-out process

1. Freeze Runtime and Evaluator version.
2. Collect 60–80 outputs from real runs, controlled failures and different writing styles.
3. Hide all Evaluator predictions from annotators.
4. Label `passed`, `action`, dimension scores, issue codes and rationale.
5. Prefer two independent annotators and adjudicate disagreements.
6. Seal gold labels before running the Evaluator.
7. Report pass precision, action macro-F1, critical-issue recall, slice metrics and bootstrap intervals.
8. If rules are changed after error analysis, the old held-out set becomes a development set; create a new held-out version.

## End-to-end metrics

For the full Agent, report task completion rate, first-pass rate, recovery success, replan success, final human acceptance, average wall time and model calls. Keep process success separate from human quality acceptance.
