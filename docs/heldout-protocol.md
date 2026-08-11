# Held-out 评测操作规程

目标是产生可以在简历和面试中解释、复现且不夸大的 Agent 指标。公开开发集用于调试，私有 held-out 集只用于冻结后的最终测量，两者不能混用。

## 1. 当前开发阶段

`evals/e2e/dev.jsonl` 是公开开发集。它覆盖直接对话、调研、比较、约束报告、多轮指代、Planner 边界、Evaluator 和后台可靠性。可以使用它修改代码，因此其分数不能写成 held-out 指标。

```powershell
durable-agent dataset-validate
durable-agent e2e-run --mode deterministic --output results/e2e/dev-check
```

## 2. 由人工独立编写 held-out

运行以下命令只生成空白工作表；它位于 `evals/private/`，默认不会被 Git 提交：

```powershell
durable-agent heldout-init --count 24
```

建议 24 条任务按以下比例编写：

- 8 条 direct：解释、追问、普通交流，以及包含“调研”等词但明确不要求执行任务的否定请求；
- 6 条单主题或多主题 research/report；
- 3 条 comparison；
- 3 条 constrained report；
- 4 条 multi-turn，其中至少两条包含指代或对上一轮约束的继承。

编写原则：

1. 使用真实用户口吻，不复制或同义改写开发集句子。
2. 在运行 Agent 之前写完 `expected` 和 `human_review`。
3. `required_topics` 只写任务真正必须覆盖的概念，不能看到输出后再补。
4. task 必须要求 `final_status=done`、合法引用和最低证据数。
5. direct 删除 task 专属字段，避免用不存在的工作流状态评分。
6. 不把模板或最终 held-out 文件提交到公开仓库。

填写完毕后运行：

```powershell
durable-agent dataset-validate --dataset evals/private/heldout-v1.jsonl
```

## 3. 冻结代码、数据和 rubric

确定不再根据 held-out 修改代码后创建锁文件：

```powershell
durable-agent eval-freeze `
  --dataset evals/private/heldout-v1.jsonl `
  --rubric evals/e2e/rubric.md `
  --output results/frozen/heldout-v1.lock.json

durable-agent eval-verify results/frozen/heldout-v1.lock.json
```

锁文件记录 dataset SHA256、rubric SHA256、关键 Agent 代码 SHA256、包版本和 Git commit。正式运行必须传入 `--lock`；任何关键文件变化都会停止运行。

## 4. 正式 LLM 运行

24 条任务重复 3 次，共 72 次运行。该步骤会产生 API 成本，执行前应确认模型、温度、环境变量和预算。

```powershell
durable-agent e2e-run `
  --dataset evals/private/heldout-v1.jsonl `
  --lock results/frozen/heldout-v1.lock.json `
  --mode llm `
  --repeats 3 `
  --output results/final/heldout-v1
```

同一冻结版本再运行 `single_pass`、`no_quality_loop` 和 `full`，用于报告质量闭环带来的差异。

## 5. 盲法人工审核

生成的审核文件不包含 Evaluator action 和分数：

```powershell
durable-agent review-pack `
  results/final/heldout-v1/runs.jsonl `
  --output results/final/heldout-v1/reviewer-a.jsonl
```

审核人只看请求、答案和预先定义的语义标准，填写：

- `accepted`: 整体是否可接受；
- `critical_issue`: 是否存在会使结果不可用的关键错误；
- `reason`: 简短依据；
- `reviewer_id`: 审核人标识。

尽可能由两人独立审核。出现分歧时另存 adjudicated 文件，不能让任一审核人看到 Evaluator 预测后再修改原始标签。

## 6. 最终可报告指标

至少分别报告：

- 自动 rubric 任务完成率；
- 人工最终接受率；
- 首次通过率和重规划成功率；
- 引用合法率；
- 故障注入恢复成功率和重复执行率；
- 平均耗时；
- full 相对 single-pass / no-quality-loop 的提升；
- 样本数、重复次数、模型版本和 95% bootstrap 置信区间。

如果看过 held-out 结果后修改 Agent、Evaluator、prompt 或评分规则，该 held-out 立即降级为开发集，最终指标必须使用一套新的未见测试集。
