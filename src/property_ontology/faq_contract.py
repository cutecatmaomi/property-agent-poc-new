"""Shared Day 1/2 FAQ contract and version inputs. No approval or external action capability."""

import copy
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .domain import digest, scoped

FAQ_FORMAT = "day1-qa-supplement-v2"
CATEGORY_SERVICE = {
    "ACTIVITY": "activity", "MEMBERSHIP": "membership", "COMPLAINT": "complaint",
    "RENOVATION": "renovation", "PROPERTY_FEE": "property_fee", "PARKING": "parking",
    "WARRANTY": "warranty", "REPAIR": "maintenance", "DELIVERY": "delivery", "PURCHASE": "purchase",
    "LOCAL_INFORMATION": "city_info", "ACCESS": "property_service", "PROJECT_FACT": "project_info",
}
CATEGORY_LABELS = {
    "ACTIVITY": "活动查询与报名指引", "MEMBERSHIP": "会员积分", "COMPLAINT": "投诉建议", "RENOVATION": "装修管理",
    "PROPERTY_FEE": "物业收费及维修资金", "PARKING": "停车与充电", "WARRANTY": "保修责任", "REPAIR": "维修报修",
    "DELIVERY": "交付办理", "PURCHASE": "购房与产权", "LOCAL_INFORMATION": "城市及周边信息",
    "ACCESS": "日常物业服务", "PROJECT_FACT": "项目基础信息",
}
KNOWLEDGE_SERVICES = frozenset(CATEGORY_SERVICE.values())
FAQ_VERSION_FIELDS = (
    "contract_version", "estate_id", "question", "answer_candidate", "reviewed_answer", "category",
    "service_code", "service_id", "source_file", "source_sha256", "source_locator", "source_rows",
    "source_question", "source_answer", "source_work_order", "document_id", "chunk_id", "scope_status",
    "verification_status", "blocking_issues", "valid_from", "valid_to",
    "applicable_to",
)
SEMANTIC_REVIEW_FIELDS = (
    "confirmed_conditions", "proposed_answer", "proposed_valid_from", "proposed_valid_to",
    "source_check", "scope_check", "conflict_check",
)


def faq_version(record):
    payload = {k: record[k] for k in FAQ_VERSION_FIELDS}
    payload["human_review"] = {k: record["human_review"][k] for k in SEMANTIC_REVIEW_FIELDS}
    return digest(payload)


def empty_human_review():
    return {
        "suggestion": "待审核", "reviewer": "", "note": "", "confirmed_conditions": "",
        "proposed_answer": "", "proposed_valid_from": None, "proposed_valid_to": None,
        "source_check": "待核验", "scope_check": "待核验", "conflict_check": "待核验",
    }


def xlsx_chunk_identity(scope, source_sha256, locator, question, answer):
    document_id = f"{scope}:DOC:XLSX:{source_sha256}"
    text = question + "\n" + answer
    identity = [document_id, locator, digest(text.encode()), "native_xlsx", FAQ_FORMAT]
    return document_id, f"{scope}:CHUNK:{digest(identity)}", digest(identity), text


class Exact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class XlsxLocator(Exact):
    kind: Literal["XLSX_CELLS"]
    sheet: str = Field(min_length=1)
    row: int = Field(ge=2, strict=True)
    question_cell: str
    answer_cell: str
    metadata_cells: list[str]

    @model_validator(mode="after")
    def matching_cells(self):
        if (self.question_cell, self.answer_cell, self.metadata_cells) != (
            f"A{self.row}", f"B{self.row}", [f"C{self.row}", f"D{self.row}"]
        ):
            raise ValueError("XLSX行号与单元格不一致")
        return self


class SourceWorkOrder(Exact):
    flag: Literal["是", "否"]
    type: str


class FAQApplicability(Exact):
    estate_id: Literal["E001", "E002"]
    residence_category: str | None
    has_elevator: bool | None
    building_form: str | None
    renovation_activity: Literal["BARE", "MINOR", "DEMOLITION", "FLOOR_HEATING"] | None


def general_applicability(scope):
    return {"estate_id": scope, "residence_category": None, "has_elevator": None,
            "building_form": None, "renovation_activity": None}


class FAQHumanReview(Exact):
    suggestion: Literal["待审核", "建议通过", "需修改", "拒绝"]
    reviewer: str
    note: str
    confirmed_conditions: str
    proposed_answer: str
    proposed_valid_from: str | None
    proposed_valid_to: str | None
    source_check: Literal["待核验", "已核验", "有差异"]
    scope_check: Literal["待核验", "已核验", "有差异"]
    conflict_check: Literal["待核验", "已解决", "有冲突"]

    @model_validator(mode="after")
    def valid_proposed_dates(self):
        start = date.fromisoformat(self.proposed_valid_from) if self.proposed_valid_from else None
        end = date.fromisoformat(self.proposed_valid_to) if self.proposed_valid_to else None
        if end and (not start or end < start):
            raise ValueError("拟定有效期区间无效")
        return self


class FAQRecord(Exact):
    contract_version: Literal["day1-qa-supplement-v2"]
    estate_id: Literal["E001", "E002"]
    faq_id: str
    version_hash: str
    question: str = Field(min_length=1)
    answer_candidate: str = Field(min_length=1)
    reviewed_answer: str | None
    category: str
    service_code: str
    service_id: str
    scope_status: Literal["PROJECT_DETAILS_UNCONFIRMED", "GENERAL_QUESTION_SCOPE_CONFIRMED", "VERIFIED"]
    review_status: Literal["PENDING_REVIEW", "APPROVED", "REJECTED", "RETIRED"]
    verification_status: Literal["PENDING_VERIFICATION", "VERIFIED"]
    valid_from: str | None
    valid_to: str | None
    reviewed_hash: str | None
    review_decision_id: str | None
    runtime_imported: bool
    executable_action: Literal["NONE"]
    source_work_order: SourceWorkOrder
    blocking_issues: list[str]
    source_file: str
    source_sha256: str
    source_rows: list[int] = Field(min_length=1)
    source_locator: XlsxLocator
    document_id: str
    chunk_id: str
    source_question: str
    source_answer: str
    baseline_comparison_ids: list[str]
    matching_original_evidence: list[dict]
    human_review: FAQHumanReview
    applicable_to: FAQApplicability
    is_mock: Literal[False]
    data_kind: Literal["REAL_DOCUMENT_QA_CANDIDATE"]

    @model_validator(mode="after")
    def exact_version_and_scope(self):
        data = self.model_dump()
        if CATEGORY_SERVICE.get(self.category) != self.service_code:
            raise ValueError("FAQ分类与服务编码不一致")
        if self.service_id != f"{self.estate_id}:SVC:{self.service_code}":
            raise ValueError("FAQ服务映射错误")
        if self.applicable_to.estate_id != self.estate_id:
            raise ValueError("FAQ适用条件跨楼盘")
        if any(not scoped(data[k], self.estate_id) for k in ("faq_id", "service_id", "document_id", "chunk_id")):
            raise ValueError("FAQ对象跨楼盘")
        if any(not scoped(k, self.estate_id) for k in self.baseline_comparison_ids):
            raise ValueError("FAQ对照引用跨楼盘")
        if any(not scoped(e.get("chunk_id"), self.estate_id) for e in self.matching_original_evidence):
            raise ValueError("原件片段对照跨楼盘")
        if self.source_locator.row not in self.source_rows or len(set(self.source_rows)) != len(self.source_rows):
            raise ValueError("FAQ原始行号集合无效")
        if self.answer_candidate not in self.source_answer:
            raise ValueError("候选答复不是连续原文")
        doc_id, chunk_id, _, _ = xlsx_chunk_identity(
            self.estate_id, self.source_sha256, data["source_locator"], self.source_question, self.source_answer
        )
        if (self.document_id, self.chunk_id) != (doc_id, chunk_id):
            raise ValueError("FAQ原件与证据标识不一致")
        expected = faq_version(data)
        if self.version_hash != expected or self.faq_id != f"{self.estate_id}:FAQ:{expected}":
            raise ValueError("FAQ内容、分类、条件或来源变化后必须生成新版本")
        if self.valid_from:
            start = date.fromisoformat(self.valid_from)
            if self.valid_to and date.fromisoformat(self.valid_to) < start:
                raise ValueError("FAQ真实有效期无效")
        elif self.valid_to:
            raise ValueError("FAQ缺少有效起日")
        if self.review_status == "APPROVED" and (
            self.verification_status != "VERIFIED" or self.scope_status != "VERIFIED"
            or self.blocking_issues or not self.valid_from or not self.reviewed_answer
            or self.reviewed_hash != self.version_hash or not self.review_decision_id
            or not scoped(self.review_decision_id, self.estate_id)
            or not self.human_review.confirmed_conditions.strip()
            or self.human_review.suggestion != "建议通过" or not self.human_review.reviewer.strip()
            or self.human_review.source_check != "已核验" or self.human_review.scope_check != "已核验"
            or self.human_review.conflict_check != "已解决"
            or self.human_review.proposed_valid_from != self.valid_from
            or self.human_review.proposed_valid_to != self.valid_to
            or self.reviewed_answer != (self.human_review.proposed_answer or self.answer_candidate)
        ):
            raise ValueError("FAQ缺少精确版本人审、范围、来源或有效期核验")
        return self


def prepare_review_revision(original, human_review):
    """Generate a new pending version from review proposals. Cannot approve or publish."""
    FAQRecord.model_validate(original)
    proposal = FAQHumanReview.model_validate(human_review).model_dump()
    revised = copy.deepcopy(original)
    revised.update(review_status="PENDING_REVIEW", verification_status="PENDING_VERIFICATION",
                   reviewed_answer=None, reviewed_hash=None, review_decision_id=None,
                   valid_from=None, valid_to=None, runtime_imported=False, human_review=proposal)
    revised["version_hash"] = faq_version(revised)
    revised["faq_id"] = f"{revised['estate_id']}:FAQ:{revised['version_hash']}"
    FAQRecord.model_validate(revised)
    return revised
