# Tool Boundaries and Agent Safety

Agent工具的安全边界用于阻止权限越界：工具注册表定义名称、参数结构、风险级别和处理函数，Runtime 再按任务类型暴露白名单或允许集合。模型不能通过文本增加新工具、绕过授权与参数校验、修改 Graph 边或提升重试预算；未知或未授权调用应被拒绝并记录为失败。

Tools are registered with a name, description, argument schema, risk class, and handler. The Agent receives only schemas allowed for the current task. Research tasks may call `search_sources`; analysis and report tasks synthesize dependency results without arbitrary tool access.

The program validates the requested tool name and allowed set before execution. A model request for an unknown or unauthorized tool fails as data rather than becoming an unrestricted function call. The Agent Loop also has a finite step budget and caches identical tool calls within one task.

Tool outputs and conversation content are treated as untrusted data in prompts. This boundary reduces prompt-injection authority: retrieved text can inform an answer, but it cannot register a new tool, alter Graph edges, increase retry budgets, or bypass deterministic citation checks.
