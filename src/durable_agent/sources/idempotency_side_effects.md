---
title: Idempotency and External Side Effects
aliases: [idempotency, idempotent, 幂等, 幂等性, 重复执行]
domains: [durable execution, external side effects]
related: [checkpoint recovery, unique constraint, delivery record]
confusable_with: [retry, exactly-once execution]
---
# Idempotency and External Side Effects

幂等表示同一逻辑请求重复执行时不会产生额外的业务效果。检查点恢复可能重复经过崩溃附近的代码，因此外部 API、消息、邮件、支付和文件写入仍需稳定幂等键、唯一约束或投递记录。

An operation is idempotent when repeating the same logical request has the same intended effect as applying it once. Agent recovery may repeat code around a crash boundary, so database writes, messages, payments, emails, and remote API calls need their own idempotency design.

A common pattern assigns a stable idempotency key to a logical effect and stores a uniqueness constraint or delivery record. Before repeating the effect, the program checks whether that key was already committed. This closes the gap between workflow checkpoint persistence and an external system that has separate transaction boundaries.

This project applies the pattern to background conversation delivery. A final report uses `job:<job_id>` as `source_id`; the Conversation Store has a unique index, so a recovered Worker returns the existing message instead of appending a duplicate report.
