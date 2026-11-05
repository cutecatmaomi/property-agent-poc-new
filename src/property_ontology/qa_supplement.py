"""Read-only XLSX extraction and scoped Day 1 candidate preparation. Never publishes knowledge."""

import copy
import json
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from functools import lru_cache
from io import BytesIO
from pathlib import PurePosixPath
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from .domain import digest
from .faq_contract import (
    FAQ_FORMAT,
    FAQRecord,
    empty_human_review,
    faq_version,
    general_applicability,
    xlsx_chunk_identity,
)

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
FORMAT = FAQ_FORMAT


def normalized(value):
    # Decimal points, units, minus signs and conditions remain significant.
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))


def question_key(value):
    return normalized(value).rstrip("?？。！!")


def extract_xlsx(path):
    """Read XML rows rather than trusting the workbook's incorrect A1 dimension."""
    # Always read current bytes. Only identical source parsing is reused; scope,
    # source hashes, reviews and all business applicability gates stay live.
    return copy.deepcopy(_xlsx_rows(path.read_bytes()))


@lru_cache(maxsize=8)
def _xlsx_rows(raw):
    rows = []
    source_hash = digest(raw)
    with ZipFile(BytesIO(raw)) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = [
                "".join(t.text or "" for t in item.findall(".//s:t", NS))
                for item in ET.fromstring(archive.read("xl/sharedStrings.xml"))
            ]
        relations = {
            r.get("Id"): r.get("Target")
            for r in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        }
        for sheet in ET.fromstring(archive.read("xl/workbook.xml")).findall("s:sheets/s:sheet", NS):
            rid = sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
            target = relations[rid]
            member = target.lstrip("/") if target.startswith("/") else str(PurePosixPath("xl") / target)
            data = ET.fromstring(archive.read(member))
            for row in data.findall("s:sheetData/s:row", NS):
                values = {}
                for cell in row.findall("s:c", NS):
                    if cell.find("s:f", NS) is not None:
                        raise ValueError("Q&A原表包含公式，需人工核验后再抽取")
                    raw = cell.find("s:v", NS)
                    value = raw.text or "" if raw is not None else ""
                    if cell.get("t") == "s" and value:
                        value = strings[int(value)]
                    elif cell.get("t") == "inlineStr":
                        value = "".join(t.text or "" for t in cell.findall("s:is//s:t", NS))
                    values[re.sub(r"\d+$", "", cell.get("r"))] = value
                number = int(row.get("r"))
                if number == 1:
                    if [values.get(k) for k in "ABCD"] != ["问题描述", "答案", "是否关联工单", "关联工单类型"]:
                        raise ValueError("Q&A表头变化，停止自动列映射")
                    continue
                if not any(values.values()):
                    continue
                if not values.get("A") or not values.get("B"):
                    raise ValueError(f"{sheet.get('name')}第{number}行缺问题或答案")
                rows.append(
                    {
                        "source_row_id": f"XLSX:{source_hash[:12]}:{sheet.get('name')}:{number}",
                        "sheet": sheet.get("name"), "row": number,
                        "question": values["A"], "answer": values["B"],
                        "work_order_flag": values.get("C", ""), "work_order_type": values.get("D", ""),
                        "source_sha256": source_hash,
                        "locator": {"kind": "XLSX_CELLS", "sheet": sheet.get("name"), "row": number,
                                    "question_cell": f"A{number}", "answer_cell": f"B{number}",
                                    "metadata_cells": [f"C{number}", f"D{number}"]},
                    }
                )
    return rows


def deduplicate(rows):
    unique, aliases = {}, {}
    for row in rows:
        key = tuple(normalized(row[k]) for k in ("question", "answer", "work_order_flag", "work_order_type"))
        if key in unique:
            unique[key]["source_rows"].append(row["row"])
            aliases[row["row"]] = unique[key]["row"]
        else:
            unique[key] = {**row, "source_rows": [row["row"]]}
    return list(unique.values()), aliases


def category(question):
    groups = [
        ("ACTIVITY", "activity", r"活动|报名|抽奖|礼品"),
        ("MEMBERSHIP", "membership", r"万象星|积分|老带新|佣金|推荐亲"),
        ("COMPLAINT", "complaint", r"投诉|不满意|泄露|骚扰"),
        ("RENOVATION", "renovation", r"装修|拆除|封窗|改动|改造|移位|更改.*管|押金|烟感.*拆|消防.*拆"),
        ("PROPERTY_FEE", "property_fee", r"物业费|物管费|管理费|物业服务费|维修基金|维修资金|公.*能耗"),
        ("PARKING", "parking", r"车位|车库|停车|行车|充电|电动车|车棚"),
        ("WARRANTY", "warranty", r"质保|保修|维保.*区别|维保服务中心是"),
        ("REPAIR", "maintenance", r"报修|维修|整改|划痕|划伤|空鼓|开裂|渗漏|松|损坏|修复|脱落|堵塞|漏水|破损"),
        ("DELIVERY", "delivery", r"交付|收房|收楼|验房|钥匙|验收单|面积补差"),
        ("PURCHASE", "purchase", r"购房|合同|发票|产证|产权证|售楼|在售|房源|房价|退款|定金|置业顾问"),
        ("LOCAL_INFORMATION", "city_info", r"学校|学区|小学|中学|幼儿园|医院|地铁|商场|落户|入托|周边"),
        ("ACCESS", "property_service", r"门禁|访客|搬家|出租|租赁|开通|缴费|邮件|邮包|管家|部门|电话|服务中心"),
    ]
    for name, code, pattern in groups:
        if re.search(pattern, question):
            return name, code
    return "PROJECT_FACT", "project_info"


def quality_issues(row):
    q, a = row["question"], row["answer"]
    result = ["HUMAN_REVIEW_REQUIRED", "POLICY_VALIDITY_UNCONFIRMED"]
    if re.search(r"填[写表].{0,12}工单|下方.{0,15}工单|专人.{0,8}联系|工作人员联系|已经安排|我现在联系|安排了专业", a):
        result.append("CALLBACK_OR_WORKORDER_PROMISE_REWRITE_REQUIRED")
    if row["work_order_flag"] == "是":
        result.append("SOURCE_WORK_ORDER_HINT_IS_NOT_ACTION_AUTHORIZATION")
    if re.search(r"身份证|银行卡|联系方式|手机号|联系地址|姓名", a):
        result.append("PERSONAL_DATA_COLLECTION_WORKFLOW_NOT_IMPLEMENTED")
    if re.search(r"规定|条例|办法|规范|合同第|合同补充|法律|政府.*文件|标准.*GB", a):
        result.append("REFERENCED_RULE_OR_CONTRACT_NEEDS_VERIFICATION")
    if re.search(r"待定|暂定|预计|规划中|以后期|具体以|以实际|视.*确定", a):
        result.append("TENTATIVE_OR_CONDITIONAL_CONTENT")
    if re.search(r"(?:20\d{2}年|今年|今冬|本次交付|集中交付|开园|完工|在售)", a):
        result.append("HISTORICAL_OR_TIME_SENSITIVE")
    if re.search(r"答[:：]|后续楼座施工|配套篇|统一说辞|生活配套类", a):
        result.append("MULTIPLE_TOPICS_IN_ONE_CELL")
    if re.search(r"(?:\d[:：]|层高\s*米|20\d{2}年[月日]|___|：[；;]|服务费[:：][；;])", a) or a.rstrip().endswith(("是", "为", "-", "—", "：", ":", "是\n")):
        result.append("TRUNCATED_OR_PLACEHOLDER_CONTENT")
    if len(q.strip()) < 9 or q.startswith("中：") or q == "哪个部门处理？":
        result.append("QUESTION_CONTEXT_MISSING")
    if re.search(r"防护|栏杆|消防|承重|梁柱|燃气|安全|电梯|辐射|甲醛|空气质量", q + a):
        result.append("SAFETY_CLAIM_REQUIRES_QUALIFIED_REVIEW")
    if re.search(r"大厦|悦府|华润城|灵山湾|崂山|云府|胶州|市南区|莱山区|芝罘区|烟台|青岛|黄岛", q + a):
        result.append("LOCATION_OR_PROJECT_CONTEXT_NEEDS_REVIEW")
    return result


ISSUE_LABELS = {
    "HUMAN_REVIEW_REQUIRED": "新增内容待人工审核",
    "POLICY_VALIDITY_UNCONFIRMED": "真实有效期未确认",
    "CALLBACK_OR_WORKORDER_PROMISE_REWRITE_REQUIRED": "含建单/回电/已安排承诺，需按当前功能重写",
    "SOURCE_WORK_ORDER_HINT_IS_NOT_ACTION_AUTHORIZATION": "原表工单关联仅为来源提示",
    "PERSONAL_DATA_COLLECTION_WORKFLOW_NOT_IMPLEMENTED": "含个人资料收集，当前未实现相应流程",
    "REFERENCED_RULE_OR_CONTRACT_NEEDS_VERIFICATION": "引用规范/合同/政策需核验原件及现行性",
    "TENTATIVE_OR_CONDITIONAL_CONTENT": "含暂定/预计/条件限定",
    "HISTORICAL_OR_TIME_SENSITIVE": "含历史日期或时效性表述",
    "MULTIPLE_TOPICS_IN_ONE_CELL": "单元格混入多个问答/其他主题",
    "TRUNCATED_OR_PLACEHOLDER_CONTENT": "答案疑似截断或留有占位符",
    "QUESTION_CONTEXT_MISSING": "问题缺上下文或疑似残片",
    "SAFETY_CLAIM_REQUIRES_QUALIFIED_REVIEW": "安全/质量断言须专业复核",
    "LOCATION_OR_PROJECT_CONTEXT_NEEDS_REVIEW": "含城市/楼盘特定上下文",
    "PROJECT_DETAILS_UNCONFIRMED": "具体项目参数不得直接套用本楼盘",
    "SAME_QUESTION_DIFFERENT_ANSWERS": "同题不同答，需核对项目/分期/业态",
    "BASELINE_CONFLICT_REVIEW_REQUIRED": "与Day1资料存在差异，禁止覆盖旧值",
    "PARTIAL_OVERLAP_WITH_BASELINE": "部分已有知识重合，保留补充部分审核",
    "POSSIBLE_DUPLICATE_NEEDS_REVIEW": "问题文字相似，需人工确认是否重复",
}


def prepare_candidates(rows, policies, evidence, decisions, source_file):
    unique, aliases = deduplicate(rows)
    question_answers = defaultdict(set)
    for r in unique:
        question_answers[question_key(r["question"])].add(normalized(r["answer"]))
    candidates, comparisons, skipped = [], [], []
    by_estate = {e: [p for p in policies if p["estate_id"] == e] for e in ("E001", "E002")}
    for row in unique:
        for scope in ("E001", "E002"):
            decision = decisions.get(f"{scope}:{row['row']}", {})
            # Explicitly named third-party project contact details are not generic property knowledge.
            if 49 <= row["row"] <= 57 and row["row"] != 56:
                decision = {"disposition": "OUT_OF_SCOPE", "note": "明确为其他楼盘专属电话，不移植"}
            if row["row"] == 56 and scope == "E001":
                decision = {"disposition": "OUT_OF_SCOPE", "note": "明确为E002电话，不移植至E001"}
            text = row["question"] + row["answer"]
            if re.search(r"胶州华润城|悦府|灵山湾|崂山|青岛华润城", text):
                decision = {"disposition": "OUT_OF_SCOPE", "note": "明确指向其他楼盘，不把项目事实迁移到E001/E002"}
            if scope == "E002" and "港城华府" in text:
                decision = {"disposition": "OUT_OF_SCOPE", "note": "明确为E001项目内容，不移植至E002"}
            refs = []
            for title in decision.get("baseline_titles", []):
                refs.extend(p["policy_id"] for p in by_estate[scope] if p["title"] == title)
                if not any(p["title"] == title for p in by_estate[scope]):
                    raise ValueError("人工对照配置引用不存在的知识标题: " + title)
            cmp = {
                "estate_id": scope, "source_row": row["row"], "source_rows": row["source_rows"],
                "question": row["question"], "baseline_ids": refs,
                "disposition": decision.get("disposition", "NEW_REVIEW"), "note": decision.get("note", ""),
            }
            if cmp["disposition"] in {"COVERED_EXISTING", "OUT_OF_SCOPE"}:
                skipped.append(cmp)
                comparisons.append(cmp)
                continue
            issues = quality_issues(row)
            if len(question_answers[question_key(row["question"])]) > 1:
                issues.append("SAME_QUESTION_DIFFERENT_ANSWERS")
            if decision.get("conflict"):
                issues.append("BASELINE_CONFLICT_REVIEW_REQUIRED")
            if refs:
                issues.append("PARTIAL_OVERLAP_WITH_BASELINE")
            cat, service = category(row["question"])
            detail_specific = cat in {"PROJECT_FACT", "LOCAL_INFORMATION"} or "LOCATION_OR_PROJECT_CONTEXT_NEEDS_REVIEW" in issues or bool(
                re.search(r"\d+(?:\.\d+)?\s*(?:元|米|㎡|m|M|户|栋|层|号楼|%|天|日|年|月)|053\d-|\d+#|A区|B区|地块|售楼处|前台|中心位于", row["question"] + row["answer"])
            )
            if detail_specific:
                issues.append("PROJECT_DETAILS_UNCONFIRMED")
            question = decision.get("question", row["question"])
            answer = decision.get("answer_excerpt", row["answer"])
            if answer not in row["answer"]:
                raise ValueError("补充答案必须是源单元格的连续原文，不自动编造或改写")
            document_id, chunk, _, _ = xlsx_chunk_identity(scope, row["source_sha256"], row["locator"], row["question"], row["answer"])
            baseline = sorted(
                by_estate[scope],
                key=lambda p: SequenceMatcher(None, question_key(row["question"]), question_key(p["question"])).ratio(),
                reverse=True,
            )
            closest = baseline[0]
            similar = SequenceMatcher(None, question_key(row["question"]), question_key(closest["question"])).ratio()
            overlap = refs or ([closest["policy_id"]] if similar >= 0.58 else [])
            if overlap and not refs:
                issues.append("POSSIBLE_DUPLICATE_NEEDS_REVIEW")
            original_sources = [
                {"chunk_id": e["id"], "source_file": e["source_path"], "locator": json.loads(e["locator_json"]),
                 "is_ocr": e["extraction_method"].lower().find("ocr") >= 0}
                for e in evidence[scope]
                if len(normalized(answer)) > 18 and normalized(answer) in normalized(e["text"])
            ]
            scope_status = "PROJECT_DETAILS_UNCONFIRMED" if detail_specific else "GENERAL_QUESTION_SCOPE_CONFIRMED"
            candidate = {
                "contract_version": FORMAT, "estate_id": scope,
                "question": question, "answer_candidate": answer, "reviewed_answer": None,
                "category": cat, "service_code": service, "service_id": f"{scope}:SVC:{service}",
                "scope_status": scope_status,
                "review_status": "PENDING_REVIEW", "verification_status": "PENDING_VERIFICATION",
                "valid_from": None, "valid_to": None, "reviewed_hash": None,
                "review_decision_id": None, "runtime_imported": False, "executable_action": "NONE",
                "source_work_order": {"flag": row["work_order_flag"], "type": row["work_order_type"]},
                "blocking_issues": sorted(set(issues)), "source_file": source_file,
                "source_sha256": row["source_sha256"], "source_rows": row["source_rows"],
                "source_locator": row["locator"], "document_id": document_id,
                "chunk_id": chunk, "source_question": row["question"], "source_answer": row["answer"],
                "baseline_comparison_ids": overlap, "matching_original_evidence": original_sources,
                "human_review": empty_human_review(),
                "applicable_to": general_applicability(scope),
                "is_mock": False, "data_kind": "REAL_DOCUMENT_QA_CANDIDATE",
            }
            candidate["version_hash"] = faq_version(candidate)
            faq_id = candidate["faq_id"] = f"{scope}:FAQ:{candidate['version_hash']}"
            candidates.append(candidate)
            cmp.update(faq_id=faq_id, baseline_ids=overlap, issues=candidate["blocking_issues"], original_evidence_count=len(original_sources))
            comparisons.append(cmp)
    return {"candidates": candidates, "comparisons": comparisons, "skipped": skipped,
            "source_rows": rows, "unique_rows": unique, "duplicate_row_map": aliases,
            "same_question_multiple_answers": sum(len(a) > 1 for a in question_answers.values())}


def validate_prepared_candidate(candidate):
    """Extraction output gate: no edited status, cross-estate reference or untraceable answer."""
    c = candidate
    FAQRecord.model_validate(c)
    scope = c["estate_id"]
    if scope not in {"E001", "E002"} or any(
        not c[k].startswith(scope + ":") for k in ("faq_id", "service_id", "document_id", "chunk_id")
    ) or any(not key.startswith(scope + ":") for key in c["baseline_comparison_ids"]):
        raise ValueError("候选引用跨楼盘")
    if (c["review_status"] != "PENDING_REVIEW" or c["verification_status"] != "PENDING_VERIFICATION"
            or c["executable_action"] != "NONE" or c["runtime_imported"] is not False
            or any(c[k] is not None for k in ("reviewed_answer", "reviewed_hash", "review_decision_id", "valid_from", "valid_to"))):
        raise ValueError("提取器不能审核、发布或授权业务动作")
    if c["answer_candidate"] not in c["source_answer"] or not c["blocking_issues"]:
        raise ValueError("候选答案不能离开来源或绕过核验")
    if c["service_id"] != f"{scope}:SVC:{c['service_code']}":
        raise ValueError("服务引用不一致")
    expected = faq_version(c)
    if c["version_hash"] != expected or c["faq_id"] != f"{scope}:FAQ:{expected}":
        raise ValueError("内容/条件/来源变化后必须生成新版本并重新审核")
