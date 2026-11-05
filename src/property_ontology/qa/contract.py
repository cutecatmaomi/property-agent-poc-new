"""Answer transport and internal candidates; these are not new Ontology entities."""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AnswerResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["ANSWERED", "NEED_CLARIFICATION", "NOT_FOUND", "CONFLICT", "NOT_VERIFIED"]
    answer_text: str = Field(min_length=1)
    matched_policy_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[dict] = Field(default_factory=list)
    estate_id: Literal["E001", "E002"] | None
    warnings: list[str] = Field(default_factory=list)
    debug_info: dict = Field(default_factory=dict)
    answer_source: Literal["TEMPLATE", "KNOWLEDGE_GENERATION", "GENERAL_CONVERSATION", "CLARIFICATION"] | None = None
    matched_template_id: str | None = None
    match_score: float | None = None
    review_status: str | None = None
    decision_reason: str = ""
    source_type: Literal["DAY1", "ADDITIONAL_QA", "GOLDEN_QUESTIONS", "ONTOLOGY_PACKAGE_POC"] | None = None
    response_mode: str | None = None
    usage_authorization: dict = Field(default_factory=dict)
    equivalence_group_id: str | None = None
    template_version_hash: str | None = None
    standard_answer: str | None = None
    generation_llm_called: bool = False
    generation_llm_call_count: int = 0
    knowledge_set: str | None = None
    quote_notice: str | None = None


@dataclass
class Candidate:
    identity: str
    estate_id: str
    version: str
    title: str
    question: str
    text: str
    service_code: str
    origin: Literal["POLICY", "FAQ"]
    facets: frozenset[str] = frozenset()
    subjects: frozenset[str] = frozenset()
    scope: dict = field(default_factory=dict)
    window: tuple = (None, None)
    fact: dict = field(default_factory=dict)
    sources: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    public_conditions: list[str] = field(default_factory=list)
    match: dict = field(default_factory=dict)
    score: float = 0
    audit: dict = field(default_factory=dict)

    def summary(self):
        return {"id": self.identity, "estate_id": self.estate_id, "version": self.version, "title": self.title,
                "scope": self.scope, "window": self.window, "fact": self.fact, "score": self.score,
                "origin": self.origin, "facets": sorted(self.facets), "subjects": sorted(self.subjects),
                "text": self.text, "notes": self.notes, "evidence_refs": self.sources, **self.audit}
