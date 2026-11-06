# Golden Questions 评测口径

`GoldenQuestions_期望答案与判定.json` 给出 100 条问题、正确楼盘、GQ ID、期望 Policy ID、版本、授权答案、原表参考答案、适用条件和证据。原始工作簿位于 `../01_原系统代码/data/knowledge_sets/GOLDEN_QUESTIONS_100/GoldenQuestions.xlsx`。评测用表保留参考答案，运行时使用已核验的授权答案。

标准题采用 `demo` 模式逐题从 `AgentSession.chat` 提交。只有同时满足以下条件才记为通过。

1. 返回 `ANSWERED` 和 `answer_source=TEMPLATE`。
2. 正确楼盘、GQ 模板 ID、Policy ID 及版本全部匹配。
3. 最终正文与 `expected_answer` 完全一致。
4. 最终答案生成模型调用次数为零，旧知识检索未进入本轮。
5. 知识咨询未创建工单。

另行统计与 `workbook_reference_answer` 的逐字一致率。E001-GQ-009 的工作簿参考文本比授权答复多一句本轮操作状态说明。授权正文、工作簿原文及差异分别保留。

同义问法需要单列语义命中率。不同楼盘、证据不足、适用条件不明确以及未通过授权的测试，应判定能否正确澄清或降级。跨楼盘串答、编造政策或绕过卡片创建工单，均计为失败。历史真实模型的结果和本地标准题结果应分栏比较。

`current_golden_regression.json` 逐题记录本包实际离线运行结果。`selected_unit_tests.json` 保存补充的编排、卡片界面、确认与权限回归统计。可从源码目录运行 `tools/enact/replay.py --all --output-dir <新目录>` 复现。
