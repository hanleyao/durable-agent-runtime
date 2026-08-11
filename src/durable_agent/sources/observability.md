# Observability and Audit Trails

控制台只显示紧凑进度，完整计划、任务调度、执行结果、评估、重规划和结束事件写入 JSONL Trace。后台队列事件、子进程日志和评测产物分别保存，以便定位不同生命周期的问题。

Compact console progress is intended for a human watching the Agent Loop. Detailed events are written to per-run JSONL traces, including plan creation, task dispatch, task execution, task results, evaluation, replanning, fault windows, and final status.

Background Job events are stored separately in SQLite because queue ownership has a different lifecycle from workflow events. Job logs capture child-process output, while status commands report queued, running, succeeded, failed, canceled, or cancel-requested states without flooding the console.

Reproducible evaluation artifacts add another layer: manifests record dataset SHA, Agent commit, version, model configuration, variant, and repeats; run records retain original inputs, routes, outputs, checks, timing, and isolated database paths. These artifacts make a reported metric auditable.
