"""One source for runtime and Day 1 FAQ schema exports; immutable baseline is never rewritten."""

import copy
import json

from .answer_selection import AnswerSelection
from .faq_contract import CATEGORY_SERVICE, FAQRecord


def runtime_schema(root):
    ROOT = root
    schema = copy.deepcopy(
        json.loads((ROOT / "outputs/schema_alignment_20261027/schema/ontology.schema.json").read_text())
    )
    schema["$id"] = "urn:property-ontology:day2-v5-source-priority"
    schema["title"] = "Day 2 runtime ontology: eleven core entities"
    schema["$defs"]["ReviewDecision"]["properties"].update(
        {
            "review_purpose": {
                "type": "string",
                "enum": ["CURRENT_POLICY", "POC_DEMO"],
                "default": "CURRENT_POLICY",
            },
            "review_batch_id": {"type": ["string", "null"]},
        }
    )
    schema["$defs"]["Policy"]["properties"].update(
        {
            "review_purpose": {"type": "string", "enum": ["CURRENT_POLICY", "POC_DEMO"]},
            "demo_reviewed_hash": {"type": "string"},
            "demo_valid_from": {"type": "string", "format": "date"},
            "demo_valid_to": {"type": ["string", "null"], "format": "date"},
            "demo_review_batch_id": {"type": "string"},
            "demo_review_decision_id": {"type": "string"},
        }
    )
    for label in ["Document", "EvidenceChunk"]:
        schema["$defs"][label]["properties"].update(
            {
                "demo_verification_status": {"type": "string", "enum": ["PENDING_VERIFICATION", "VERIFIED"]},
                "demo_reviewed_hash": {"type": "string"},
                "demo_review_batch_id": {"type": "string"},
                "demo_review_decision_id": {"type": "string"},
            }
        )
    schema["$defs"]["Unit"]["properties"]["status"] = {
        "type": "string",
        "enum": ["ACTIVE", "INACTIVE"],
        "description": "业务房屋是否可接受新报修",
    }
    schema["$defs"]["Unit"]["required"].append("status")
    schema["$defs"]["WorkOrder"]["properties"]["state_history_json"] = {
        "type": "string",
        "readOnly": True,
        "description": "后端原子写入的状态日志/持久outbox；镜像到独立SQLite日志，不增设核心对象",
    }
    work = schema["$defs"]["WorkOrder"]
    work["description"] = "住户明确确认后创建的维修报修；人工服务热线与红线回复不创建工单"
    work["properties"]["request_type"] = {"type": "string", "enum": ["REPAIR"], "default": "REPAIR"}
    collections = {
        "Estate": "estates",
        "Building": "buildings",
        "Unit": "units",
        "Resident": "residents",
        "Service": "services",
        "Policy": "policies",
        "Document": "documents",
        "EvidenceChunk": "evidence_chunks",
        "WorkOrder": "work_orders",
        "ReviewDecision": "review_decisions",
        "FAQ": "faqs",
    }
    schema["type"] = "object"
    schema["properties"] = {"schema_version": {"const": "day2-v5-source-priority"}}
    for entity, group in collections.items():
        schema["properties"][group] = {"type": "array", "items": {"$ref": "#/$defs/" + entity}}
    schema["required"] = ["schema_version", *collections.values()]

    faq = FAQRecord.model_json_schema()
    schema["$defs"].update(faq.pop("$defs", {}))
    faq["allOf"] = [
        {"if": {"properties": {"estate_id": {"const": scope}}},
         "then": {"properties": {key: {"pattern": f"^{scope}:"} for key in ("faq_id", "document_id", "chunk_id", "service_id")}}}
        for scope in ("E001", "E002")
    ] + [{
        "if": {"properties": {"review_status": {"const": "APPROVED"}}},
        "then": {"properties": {
            "reviewed_answer": {"type": "string", "minLength": 1},
            "reviewed_hash": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
            "review_decision_id": {"type": "string", "minLength": 12},
            "scope_status": {"const": "VERIFIED"}, "verification_status": {"const": "VERIFIED"},
            "valid_from": {"type": "string", "format": "date"}, "blocking_issues": {"maxItems": 0},
            "human_review": {"properties": {
                "suggestion": {"const": "建议通过"}, "reviewer": {"type": "string", "minLength": 1},
                "confirmed_conditions": {"type": "string", "minLength": 1},
                "source_check": {"const": "已核验"}, "scope_check": {"const": "已核验"},
                "conflict_check": {"const": "已解决"},
                "proposed_valid_from": {"type": "string", "format": "date"},
            }},
        }},
    }]
    faq["properties"]["category"] = {"enum": list(CATEGORY_SERVICE)}
    faq["properties"]["service_code"] = {"enum": list(CATEGORY_SERVICE.values())}
    faq["allOf"].extend({"if": {"properties": {"category": {"const": cat}}},
                         "then": {"properties": {"service_code": {"const": service}}}}
                        for cat, service in CATEGORY_SERVICE.items())
    schema["$defs"]["FAQ"] = faq
    faq["properties"].update({
        "demo_review_status": {"enum": ["APPROVED"]},
        "demo_reviewed_hash": {"type": "string"},
        "demo_valid_from": {"type": "string", "format": "date"},
        "demo_valid_to": {"type": ["string", "null"], "format": "date"},
        "demo_review_batch_id": {"type": "string"}, "demo_review_decision_id": {"type": "string"},
        "answer_selection": {"$ref": "#/$defs/AnswerSelection", "description": "独立版本的来源选择，Neo4j投影为answer_selection_json；不是第12类节点"},
        "answer_selection_hash": {"type": "string"}, "answer_selection_batch_id": {"type": "string"},
        "answer_selection_source": {"enum": ["DAY1_POLICY", "SUPPLEMENT_QA"]},
        "answer_selection_review_decision_id": {"type": "string"},
        "answer_selection_valid_from": {"type": "string", "format": "date"},
        "answer_selection_valid_to": {"type": ["string", "null"], "format": "date"},
    })
    selection = AnswerSelection.model_json_schema()
    schema["$defs"].update(selection.pop("$defs", {}))
    schema["$defs"]["AnswerSelection"] = selection
    schema["$defs"]["ReviewDecision"]["properties"]["target_component"] = {
        "enum": ["CONTENT", "ANSWER_SELECTION"], "default": "CONTENT",
        "description": "CONTENT绑定原内容版本；ANSWER_SELECTION绑定FAQ内部来源选择hash，旧内容审批不自动批准新选择"}
    schema["$defs"]["ReviewDecision"]["properties"]["target_type"]["enum"] = ["Policy", "FAQ", "Document", "EvidenceChunk"]
    schema["$defs"]["Document"]["properties"]["source_format"] = {"enum": ["PDF", "DOCX", "XLSX"]}
    schema["$defs"]["EvidenceChunk"]["properties"]["source_locator"]["description"] += "；或XLSX工作表、真实行号和单元格"
    schema["$defs"]["Service"]["properties"]["service_code"]["enum"] = list(CATEGORY_SERVICE.values())
    schema["$defs"]["Service"]["properties"]["capability_status"] = {
        "enum": ["KNOWLEDGE_ONLY", "IMPLEMENTED_ACTION", "NOT_IMPLEMENTED"]}
    schema["x-extension-relationships"] = [
        {"source": "FAQ", "relation": "APPLIES_TO", "target": "Estate", "cardinality": "1"},
        {"source": "FAQ", "relation": "GOVERNS", "target": "Service", "cardinality": "1"},
        {"source": "FAQ", "relation": "SUPPORTED_BY", "target": "EvidenceChunk", "cardinality": "1..N"},
        {"source": "ReviewDecision", "relation": "REVIEWS", "target": "FAQ", "cardinality": "1 target version"},
        {"source": "FAQ", "relation": "ANSWERED_BY", "target": "Policy", "cardinality": "0..N; same estate and exact selection_hash"},
    ]
    schema["x-invariants"] = ["all endpoints share estate_id", "no Region or inherited city policy",
                              "source work-order metadata cannot authorize actions", "no automatic approval",
                              "shared source bytes do not share approvals", "fixed replies precede FAQ retrieval"]
    schema["x-runtime-validation"] = [
        "faq_version recomputes semantic fields including category, applicability and human proposals",
        "approved FAQ requires reviewed_hash=version_hash and exact authorized CURRENT_POLICY ReviewDecision",
        "valid_from/to must equal human proposed dates; current date must lie inside interval",
        "approved FAQ additionally requires exact human verifications for Document and EvidenceChunk",
        "FAQApplicability estate and all structured conditions must match authenticated context",
        "source replay compares actual XLSX cells and bytes; JSON Schema alone does not authorize answers",
        "answer selection requires an independently versioned human ANSWER_SELECTION review",
        "Day1 selection validates each Policy version, source, scope and conditions; denial never falls back to replaced QA",
        "supplement selection displays approved source reference with unresolved conditions rather than filling gaps",
    ]
    return schema


def qa_design_schema(root):
    schema = runtime_schema(root)
    schema["$id"] = "urn:property-ontology:day1-qa-design-v3"
    schema["title"] = "Day 1/2 aligned FAQ ontology: eleven core entities"
    schema["properties"]["schema_version"] = {"const": "day1-qa-design-v3"}
    return schema
