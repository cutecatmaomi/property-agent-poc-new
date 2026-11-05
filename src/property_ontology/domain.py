"""Day 2 runtime contract. Day 1 models remain the immutable extraction format."""

import hashlib
import json
import unicodedata
from datetime import UTC, date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

EstateID = Literal["E001", "E002"]
ServiceCode = Literal[
    "property_fee", "renovation", "maintenance", "warranty", "parking", "complaint", "property_service"
]
WorkOrderStatus = Literal["SUBMITTED", "ASSIGNED", "IN_PROGRESS", "COMPLETED", "CANCELLED"]
TRANSITIONS = {
    "SUBMITTED": {"ASSIGNED", "CANCELLED"},
    "ASSIGNED": {"IN_PROGRESS", "CANCELLED"},
    "IN_PROGRESS": {"COMPLETED", "CANCELLED"},
    "COMPLETED": set(),
    "CANCELLED": set(),
}
VERSION_FIELDS = [
    "contract_version",
    "estate_id",
    "policy_family_id",
    "service_id",
    "service_code",
    "topic",
    "title",
    "question",
    "statement",
    "conditions",
    "scope_note",
    "fact",
    "provenance",
]
CORE_IDS = {
    "Estate": "estate_id",
    "Building": "building_id",
    "Unit": "unit_id",
    "Resident": "resident_id",
    "Service": "service_id",
    "Policy": "policy_id",
    "Document": "document_id",
    "EvidenceChunk": "chunk_id",
    "WorkOrder": "work_order_id",
    "ReviewDecision": "review_decision_id",
    "FAQ": "faq_id",
}
RELATIONS = {
    "HAS_BUILDING": ("Estate", "Building"),
    "HAS_UNIT": ("Building", "Unit"),
    "ASSOCIATED_WITH": ("Resident", "Unit"),
    "OFFERS": ("Estate", "Service"),
    "APPLIES_TO": ({"Policy", "FAQ"}, "Estate"),
    "GOVERNS": ({"Policy", "FAQ"}, "Service"),
    "ANSWERED_BY": ("FAQ", "Policy"),
    "SUPPORTED_BY": ({"Policy", "FAQ"}, "EvidenceChunk"),
    "HAS_CHUNK": ("Document", "EvidenceChunk"),
    "HAS_DOCUMENT": ("Estate", "Document"),
    "CREATED": ("Resident", "WorkOrder"),
    "FOR_UNIT": ("WorkOrder", "Unit"),
    "REQUESTS": ("WorkOrder", "Service"),
    "REVIEWS": ("ReviewDecision", {"Policy", "FAQ", "Document", "EvidenceChunk"}),
}


def canonical(value):
    return unicodedata.normalize(
        "NFC", json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value).encode()).hexdigest()


def utcnow():
    return datetime.now(UTC)


def estate(value):
    if value not in ("E001", "E002"):
        raise POCError("INVALID_ESTATE", "楼盘标识无效", 400)
    return value


def scoped(value, scope):
    return isinstance(value, str) and (value == scope or value.startswith(scope + ":"))


class POCError(Exception):
    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class PolicyRecord(Strict):
    # Exact alignment payload retained; nested dictionaries are hashed without semantic rewriting.
    model_config = ConfigDict(extra="forbid", frozen=True)
    contract_version: str
    estate_id: EstateID
    policy_family_id: str
    service_id: str
    service_code: ServiceCode
    topic: Literal["fee", "rule", "procedure", "contact", "hours", "responsibility", "warranty"]
    title: str
    question: str
    statement: str
    conditions: dict
    scope_note: str
    fact: dict
    provenance: list[dict]
    policy_id: str
    original_candidate_ids: list[str]
    source_file: str
    source_locator: dict
    source_excerpt: str
    document_id: str
    chunk_ids: list[str] = Field(min_length=1)
    extraction_quality: str
    status: Literal["PENDING_REVIEW", "APPROVED", "REJECTED", "RETIRED"]
    review_status: Literal["PENDING_REVIEW", "APPROVED", "REJECTED", "RETIRED"]
    review_note: str
    version_hash: str
    reviewed_hash: str | None
    valid_from: str | None
    valid_to: str | None
    validity_status: str
    conflict_status: str
    blocking_issues: list[str]
    is_mock: Literal[False]
    data_kind: Literal["REAL_DOCUMENT_POLICY_CANDIDATE"]

    @model_validator(mode="after")
    def consistent(self):
        data = self.model_dump()
        if self.status != self.review_status:
            raise ValueError("status与review_status不一致")
        if any(
            not scoped(data[k], self.estate_id)
            for k in ["policy_id", "policy_family_id", "service_id", "document_id"]
        ):
            raise ValueError("知识跨楼盘引用")
        if self.service_id != f"{self.estate_id}:SVC:{self.service_code}":
            raise ValueError("Service映射错误")
        if self.version_hash != digest({k: data[k] for k in VERSION_FIELDS}):
            raise ValueError("知识版本哈希不一致")
        if self.policy_id != f"{self.estate_id}:POL:{digest([self.policy_family_id, self.version_hash])}":
            raise ValueError("知识版本ID不一致")
        if self.conditions["applicable_to"]["estate_id"] != self.estate_id:
            raise ValueError("适用范围跨楼盘")
        return self


class RepairDraft(Strict):
    unit_id: str = Field(min_length=8, max_length=160)
    service_code: Literal["maintenance"] = "maintenance"
    description: str = Field(min_length=1, max_length=1000)
    idempotency_key: str = Field(min_length=16, max_length=128, pattern=r"^[A-Za-z0-9:_-]+$")


class CreateRequest(RepairDraft):
    confirmation_ref: str = Field(min_length=24, max_length=160)


class UpdateRequest(Strict):
    number: str = Field(min_length=6, max_length=160)
    target_status: WorkOrderStatus
    expected_version: int = Field(ge=1, strict=True)
    reason: str = Field(min_length=1, max_length=500)
    idempotency_key: str = Field(min_length=16, max_length=128, pattern=r"^[A-Za-z0-9:_-]+$")


class ReviewDecisionRecord(Strict):
    model_config = ConfigDict(extra="forbid", frozen=True)
    estate_id: EstateID
    review_decision_id: str
    actor_ref: str
    actor_role: Literal["HUMAN_REVIEWER"]
    authorization_ref: str
    target_type: Literal["Policy", "FAQ", "Document", "EvidenceChunk"]
    target_component: Literal["CONTENT", "ANSWER_SELECTION"] = "CONTENT"
    target_id: str
    target_version_hash: str
    decision: Literal["APPROVED", "REJECTED", "VERIFIED", "SUPERSEDED"]
    note: str
    decided_at: str
    version_hash: str
    is_mock: bool
    review_purpose: Literal["CURRENT_POLICY", "POC_DEMO"] = "CURRENT_POLICY"
    review_batch_id: str | None = None

    @model_validator(mode="after")
    def exact_version_and_actor(self):
        if any(
            not scoped(v, self.estate_id) for v in [self.review_decision_id, self.actor_ref, self.target_id]
        ):
            raise ValueError("审核对象与人员跨楼盘")
        # Legacy reviews did not carry a purpose field; their exact old hash remains valid.
        payload = self.model_dump(exclude={"review_decision_id", "version_hash"}, exclude_unset=True)
        if self.version_hash != digest(payload):
            raise ValueError("审核事件哈希失效")
        at = datetime.fromisoformat(self.decided_at)
        if at.tzinfo is None:
            raise ValueError("审核时间必须包含时区")
        if self.target_type in {"Policy", "FAQ"} and self.decision not in {"APPROVED", "REJECTED"}:
            raise ValueError("Policy审核决定错误")
        return self


def association_valid(a, day: date, create=False):
    try:
        start = date.fromisoformat(a["valid_from"])
        end = date.fromisoformat(a["valid_to"]) if a.get("valid_to") else date.max
        return (
            a["role"] in {"OWNER", "TENANT", "AUTHORIZED"}
            and a["status"] == "ACTIVE"
            and start <= day <= end
            and (not create or a["can_create_work_order"] is True)
        )
    except (KeyError, ValueError, TypeError):
        return False
