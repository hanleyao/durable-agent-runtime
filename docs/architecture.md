# Architecture Notes

## Reliability layers

The runtime separates five kinds of state:

1. Conversation state: complete user/assistant messages and a rolling summary keyed by `session_id`.
2. Agent context: the summary, recent messages and observations needed for the current model decision.
3. Workflow state: Task DAG, results and evaluation state persisted by LangGraph checkpoints.
4. Background job state: queue ownership, heartbeat, lease, cancellation and attempts.
5. Long-term memory: reusable knowledge retrieved across independent sessions and runs.

Mixing these stores makes recovery ambiguous. A background Job may still be running while its workflow is already terminal, and a workflow checkpoint may exist even when no Worker owns the Job.

## Conversation boundary

A conversation `session_id` is stable across turns. Every multi-step task created by that conversation receives a new LangGraph `thread_id`, so a terminal workflow never blocks the next turn. The session stores the resulting report and run ID as an assistant message.

With `chat --background`, the router still decides between `direct` and `task`, but a task route is persisted in the Job Store instead of blocking the chat process. The acknowledgement is written immediately. The Worker runs the same checkpointed Runtime in a child process and, after a successful evaluation, writes the final report back to the originating `session_id`.

Background delivery uses `job:<job_id>` as a unique message source. If a Worker crashes after inserting the report but before marking the Job terminal, recovery can repeat delivery safely: the Conversation Store returns the existing message instead of creating a duplicate. `delivered_at` and the `conversation_delivered` event provide an audit trail.

The full transcript is retained for audit. The active model context contains only a rolling summary plus recent messages. Summaries preserve goals, decisions, constraints, unresolved questions and task outcomes while dropping greetings, repetition and verbose execution logs.

A router selects `direct` for normal dialogue and `task` for research, comparison, evaluation and report requests. This avoids paying the latency and token cost of a complete DAG for every conversational turn.

## LLM and program boundary

The model proposes semantic plans and task outputs. The program owns:

- task kind allowlists;
- dependency repair and cycle rejection;
- tool registration and task-specific exposure;
- retry and evaluation budgets;
- checkpoint and Job state transitions;
- deterministic hard quality failures.

## Evaluation loop

`revise` resets the report-producing task because existing evidence is sufficient. `replan` sends the failed DAG, completed results and Evaluator feedback to the Planner, validates the replacement DAG, and reuses only completed tasks whose ID, kind, goal and dependencies are unchanged. Both actions are bounded by `max_evaluations`. `abort` terminates immediately when success cannot be defined.

## Recovery semantics

Automatic crash recovery reuses the same `thread_id` and invokes the graph from its latest checkpoint. Manual retry creates a new `thread_id`, representing a new execution rather than continuation of the old terminal state.

The three identifiers intentionally have different lifetimes:

- `session_id`: a long-lived human conversation;
- `job_id`: queue ownership, retries, cancellation and result delivery;
- `thread_id`: one checkpointed workflow execution.
