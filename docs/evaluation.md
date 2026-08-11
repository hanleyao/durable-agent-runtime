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

## Reproducible runner

`durable-agent e2e-run` executes each case with isolated conversation, memory and checkpoint databases. A run records the Agent commit, dataset SHA256, execution mode, per-turn routes, final workflow state, evidence count, evaluation action, duration and every rubric check.

The visible 12-case set is a runner-development and CI set. A final resume metric must come from a separately authored and sealed held-out set after the Agent, prompts, evaluator and rubric have been frozen.

## Baselines

The comparison runner keeps retrieval evidence constant. `single_pass` removes planning, the DAG and quality repair; `no_quality_loop` retains task execution but permits only one evaluation; `full` uses the complete runtime. This makes differences more attributable than comparing against a deliberately evidence-free prompt.

## Fault injection

The fault harness starts a real child process, waits for a traceable node boundary, forcefully kills that process and resumes the same LangGraph thread from the same SQLite checkpoint. Recovery requires a final accepted result and zero redispatches of tasks completed before the fault window. Process exit alone is not recovery.
