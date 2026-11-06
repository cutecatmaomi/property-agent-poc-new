# 报修卡片字段与确认接口

以当前 Streamlit 源码 `src/property_ontology/streamlit_app.py` 的 `agent.pending` 分支为准。卡片标题为“请确认本次报修”。

| 页面或调用字段 | 实际取值 |
|---|---|
| 房屋 | `pending.data.unit_id` 对应登录上下文的 `units[unit_id]`，展示 `building_name · room_no` |
| 故障 | `pending.data.description` |
| 方便上门的时间意向 | `pending.data.dialogue_details.available_time` 有值时展示 |
| 校验提示 | `pending.blocked_reason` 有值时提示并禁用提交按钮 |
| 确认提交报修 | 可信前端调用 `agent.confirm(pending["ticket"], accept=True)` |
| 取消本次报修 | 可信前端调用 `agent.confirm(pending["ticket"], accept=False)` |

`AgentSession.chat` 的预览返回 `kind="REPAIR_PREVIEW"`、`text`、`ticket`、`data`、`trace`。`data.kind="REPAIR_CONFIRMATION"`，含 `unit_id`、`description`、`expires_in_seconds=600`、`assistant_text`、`dialogue_details`、`dialogue_outcomes`、`knowledge_results`。票据绑定当前会话及草稿，UI 保管不透明的 `ticket`。

当前入口是 Python/Streamlit 方法。受控业务接口是 `PropertyActions.prepare_work_order`、`confirm_work_order`、`create_work_order`，字段约束见 `repair_actions_schema.json`。后台的 `confirmation_ref`、`idempotency_key` 和恢复用 nonce 由后台生成。住户发“确认”这类聊天文字时，返回按钮操作指引。确认成功才创建工单，重复确认复用原工单。用户仅提供上门时间意向时，保留其意向，实际预约仍需业务确认。

`repair_preview.json` 是本包源码走一次测试报修实际返回的结果，已遮蔽临时 ticket。原仓库的 UI 测试也在本包中复跑。
