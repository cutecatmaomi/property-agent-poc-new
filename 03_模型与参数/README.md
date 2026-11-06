# 在线模型、参数和输出约束

模型名为 `deepseek-flash`，接口为 DeepSeek 的 OpenAI 兼容 `/chat/completions`，超时 20 秒。参数来自 2026-11-05 已导出的非敏感在线配置，已用本包源码在传输前核对实际请求。

| 调用 | max_tokens | 输出 |
|---|---:|---|
| 普通规划、意图和槽位 | 1800 | 强制选择 `plan_property_turn` |
| 规划并联合匹配 | 1800 | 强制选择 `plan_property_turn`，可包含 `template_selections` |
| 独立知识语义匹配 | 1800 | 强制选择 `match_property_templates`，代码上限 2400 |
| 独立回复生成 | 1400 | 强制选择 `compose_property_reply`，代码上限 1400 |

所有在线阶段都省略 `temperature`，使用提供商默认值。请求设置 `stream=false` 和 `thinking.type=disabled`。结构化输出采用强制函数调用及 JSON 参数，服务端用 Pydantic 验证。模型请求省略 `response_format`，并使用非 strict 工具 Schema。代码接受唯一且名称正确的工具调用。

确定性安全、标准题、候选排序、授权、确认和模板直出由本地代码完成。联合规划和 inline 回复默认开启。实际部署可通过接收方自己的模型配置选择本地或在线模式。

`模型与参数.json` 保存逐阶段字段和提示词哈希。`工具协议/` 是代码实际发出的函数定义，部分字段的枚举在每轮根据当前候选和缺槽动态收窄。示例使用 E001 的 50 条目录和一个 `symptom` 缺槽。
