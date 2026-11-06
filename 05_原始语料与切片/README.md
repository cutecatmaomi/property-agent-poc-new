# 原件、切片和检索证据

`语料索引.json` 给出 8 份原件的相对路径和 SHA-256。原件放在 `../01_原系统代码/data/raw/`，保证加载器可按原路径核验文件和版本。

`实际解析切片/E001` 和 `E002` 保存原系统解析时的 `documents.jsonl`、`evidence.jsonl`、`candidates.jsonl` 和 manifest。证据切片有实际文本、文档 ID、来源路径、页码或段落位置及提取方法。候选记录包含标题、规则陈述、适用条件和原审批状态。候选阶段的事实不能直接当成已批准答案。

问答对见 `../01_原系统代码/data/knowledge_sets/GOLDEN_QUESTIONS_100/GoldenQuestions_100问答.json`。100 条均提供标准问题、原表参考回答、授权回答、关联 Policy 和证据。冻结图谱中的 Policy、EvidenceChunk、Document、ReviewDecision 保留可追溯关系。

`检索样例_当前标准题.json` 来自当前源码真实选择结果。`检索样例_历史真实在线.json` 提供原在线匹配日志中的候选数、语义选择、命中规则和证据。当前 POC 从 Golden Questions 目录选择事实，并核验冻结证据与适用条件。

如需比较 ontology 与纯切片检索，可以使用这些原件与真实解析产物建立同源对照。两路应使用相同题目、楼盘范围和事实判定，分别记录命中、误答、澄清和失败。现有样例用于检查链路，效果差异由正式对照结果判定。
