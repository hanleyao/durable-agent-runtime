# Background Jobs, Heartbeat, and Lease

后台 Job Store 管理排队、领取、心跳、租约、取消、尝试次数和 Worker 所有权。心跳续租表明 Worker 仍持有任务；租约过期允许系统回收失联任务，但不代表工作流业务状态已经丢失。

The Job Store persists queue lifecycle state independently from LangGraph workflow state. A Job records `job_id`, status, attempts, Worker identity, process ID, heartbeat time, lease expiry, cancellation request, log path, checkpoint path, and optional conversation delivery fields.

Claiming a queued Job is an atomic database transition. While running, the Worker renews a time-limited lease with heartbeats. If the lease expires, recovery either requeues the Job, marks it canceled, or fails it when the retry budget is exhausted. A lease therefore detects lost ownership; it is not the workflow checkpoint itself.

Cancellation is cooperative at the Job level but the Worker also terminates the child process tree. Automatic retry continues the same checkpoint thread after a failed attempt, while manual retry creates a new thread identity for a new execution. Logs are written to files and the console only prints compact state changes.
