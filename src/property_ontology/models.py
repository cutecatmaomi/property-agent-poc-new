"""Day 1 ontology definitions only: no database, approval or work-order operations."""

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EstateID = Literal["E001", "E002"]
Topic = Literal["service", "fee", "decoration", "repair"]
EntityLabel = Literal[
    "Estate", "Building", "Unit", "Resident", "Service", "Policy", "Document", "EvidenceChunk", "WorkOrder"
]


def now_iso():
    return datetime.now(UTC).isoformat()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def make_id(estate: EstateID, kind: str, value):
    if estate not in ("E001", "E002"):
        raise ValueError("未知楼盘")
    return f"{estate}:{kind}:{digest(value)}"


class ReviewStatus(StrEnum):
    PENDING_REVIEW = "PENDING_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    RETIRED = "RETIRED"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScopedModel(StrictModel):
    estate_id: EstateID
    id: str

    @field_validator("id")
    @classmethod
    def scoped_id(cls, value, info):
        if not value.startswith(info.data.get("estate_id", "INVALID") + ":"):
            raise ValueError("ID与楼盘不一致")
        return value

    @model_validator(mode="after")
    def scoped_references(self):
        for field, value in self.model_dump().items():
            if field.endswith("_id") and field != "estate_id" and value is not None:
                if not isinstance(value, str) or not value.startswith(self.estate_id + ":"):
                    raise ValueError("关联对象与楼盘不一致")
        return self


class Estate(StrictModel):
    estate_id: EstateID
    name: str
    status: Literal["ACTIVE", "INACTIVE"] = "ACTIVE"


class Building(ScopedModel):
    number: str
    name: str
    is_mock: bool = True


class Unit(ScopedModel):
    building_id: str
    entrance_no: str
    room_no: str
    unit_type: Literal["RESIDENTIAL", "RETAIL", "APARTMENT", "OTHER"]
    has_elevator: bool | None = None
    is_mock: bool = True


class Resident(ScopedModel):
    display_name: str
    status: Literal["ACTIVE", "INACTIVE"] = "ACTIVE"
    is_mock: Literal[True] = True


class Service(ScopedModel):
    code: str
    name: str
    category: Topic
    enabled: bool = True


class Policy(ScopedModel):
    topic: Topic
    title: str
    statement: str
    conditions: str
    document_id: str
    chunk_id: str
    version_hash: str
    status: ReviewStatus = ReviewStatus.PENDING_REVIEW
    valid_from: str | None = None
    valid_to: str | None = None
    reviewed_by: str | None = None
    reviewed_at: str | None = None


class Document(ScopedModel):
    filename: str
    source_path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    parse_status: str
    verification_status: Literal["PENDING_VERIFICATION", "VERIFIED", "REJECTED", "SUPERSEDED"] = (
        "PENDING_VERIFICATION"
    )
    issues_json: str = "[]"


class EvidenceChunk(ScopedModel):
    document_id: str
    text: str
    content_hash: str
    locator_json: str
    extraction_method: str
    quality: Literal["RELIABLE", "NEEDS_HUMAN_VERIFICATION"]
    parser_version: str


class WorkOrder(ScopedModel):
    number: str
    resident_id: str
    unit_id: str
    service_id: str
    description: str
    status: Literal["SUBMITTED", "ACCEPTED", "IN_PROGRESS", "COMPLETED", "CLOSED", "CANCELLED"] = "SUBMITTED"
    idempotency_key: str
    created_at: str
    is_mock: Literal[True] = True


RELATION_TYPES = {
    "HAS_BUILDING": {("Estate", "Building")},
    "HAS_UNIT": {("Building", "Unit")},
    "ASSOCIATED_WITH": {("Resident", "Unit")},
    "OFFERS": {("Estate", "Service")},
    "HAS_POLICY": {("Estate", "Policy")},
    "HAS_DOCUMENT": {("Estate", "Document")},
    "HAS_CHUNK": {("Document", "EvidenceChunk")},
    "SUPPORTED_BY": {("Policy", "EvidenceChunk")},
    "GOVERNS": {("Policy", "Service")},
    "APPLIES_TO": {("Policy", "Estate"), ("Policy", "Building"), ("Policy", "Unit")},
    "CREATED": {("Resident", "WorkOrder")},
    "FOR_UNIT": {("WorkOrder", "Unit")},
    "REQUESTS": {("WorkOrder", "Service")},
}


class OntologyRelation(StrictModel):
    estate_id: EstateID
    source_id: str
    source_type: EntityLabel
    relation: str
    target_id: str
    target_type: EntityLabel

    @model_validator(mode="after")
    def valid_relation(self):
        if (self.source_type, self.target_type) not in RELATION_TYPES.get(self.relation, set()):
            raise ValueError("未定义的对象关系")
        for identifier in (self.source_id, self.target_id):
            if identifier != self.estate_id and not identifier.startswith(self.estate_id + ":"):
                raise ValueError("禁止跨楼盘关系")
        return self


class OntologySnapshot(StrictModel):
    schema_version: Literal["day1-v1"] = "day1-v1"
    estates: list[Estate] = Field(default_factory=list)
    buildings: list[Building] = Field(default_factory=list)
    units: list[Unit] = Field(default_factory=list)
    residents: list[Resident] = Field(default_factory=list)
    services: list[Service] = Field(default_factory=list)
    policies: list[Policy] = Field(default_factory=list)
    documents: list[Document] = Field(default_factory=list)
    evidence_chunks: list[EvidenceChunk] = Field(default_factory=list)
    work_orders: list[WorkOrder] = Field(default_factory=list)
    relations: list[OntologyRelation] = Field(default_factory=list)


class ExtractedClaim(StrictModel):
    """Extraction never accepts estate, approval or reviewer fields."""

    topic: Topic
    title: str = Field(min_length=1, max_length=200)
    statement: str = Field(min_length=1, max_length=5000)
    conditions: str = Field(default="", max_length=1000)
