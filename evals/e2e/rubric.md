# End-to-End Evaluation Rubric

This directory is a visible development set. It may be used to debug prompts, rules and the runner, so its scores must never be presented as held-out accuracy.

## Automated completion

A run passes only when every declared automated check passes. For task routes this normally requires a terminal `done` workflow, an allowed Evaluator action, required topic coverage, sufficient evidence and valid citation markers.

`status=done` alone is not task completion. A failed rubric check makes the run incomplete even when the workflow terminated normally.

## Human review

The `human_review` field contains semantic checks that deterministic string matching cannot establish reliably. Development runs may use one reviewer. Final held-out results should use two independent reviewers where possible, hide Evaluator predictions, and adjudicate disagreements.

## Dataset roles

- Development: visible cases used while changing the system.
- Held-out: unseen cases sealed before the final run.
- Fault injection: controlled process termination trials, scored separately from answer quality.

Never combine development and held-out results into one claimed accuracy number.
