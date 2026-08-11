# Fault Injection and Recovery Testing

故障注入会在指定 Graph 边界强制终止真实子进程，再使用相同 checkpoint 和 `thread_id` 恢复。只有最终通过质量门且中断前已完成任务没有被重复调度，才算可靠恢复。

Fault injection tests recovery by starting a real child process, waiting until a trace event opens a fault window before a selected Graph node, and forcibly terminating the process. The harness then resumes with the same checkpoint database and `thread_id`.

A trial is recovered only when the fault window was observed, the process was killed, a checkpoint exists, the resumed workflow finishes successfully, and the final evaluation accepts the result. Merely restarting a command is not sufficient evidence of recovery.

The harness also inspects dispatch events to find completed tasks that were executed again after recovery. Recovery success rate and duplicate execution rate must be reported separately because a workflow can eventually succeed while still repeating expensive or externally visible work.
