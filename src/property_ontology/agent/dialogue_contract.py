"""Untrusted semantic decisions, not executable model code or policy answers."""

from typing import Literal

from pydantic import Field

from ..domain import Strict
from .planner import Service
from .reply_generation import ReplyDraft


class DialogueTask(Strict):
    intent: Literal["KNOWLEDGE", "REPAIR_GUIDANCE", "REPAIR_PREPARE", "REPAIR_CAUSE", "TRACK_ORDER", "COMPLAINT",
                    "SOCIAL", "FEEDBACK", "EXPLAIN_LIMITS", "OPERATION_STATUS"]
    source_quote: str = Field(min_length=1, max_length=300)
    query: str = Field(default="", max_length=300)
    service_code: Service | Literal["NONE"] = "NONE"
    operation_subject: Literal["CURRENT_ACTIONS", "EXTERNAL_CONTACT", "ORDER_RECEIPT"] = "CURRENT_ACTIONS"


class RepairSlots(Strict):
    location: str | None = Field(default=None, max_length=80)
    part: str | None = Field(default=None, max_length=80)
    symptom: str | None = Field(default=None, max_length=100)
    started_at: str | None = Field(default=None, max_length=80)
    available_time: str | None = Field(default=None, max_length=80)


class DialogueDecision(Strict):
    route: Literal["SERVICE", "CORRECTION", "CANCEL", "OUT_OF_SCOPE", "HUMAN_REQUEST", "RED_LINE", "CLARIFY"]
    tasks: list[DialogueTask] = Field(default_factory=list, max_length=4)
    repair: RepairSlots = Field(default_factory=RepairSlots)
    requires_free: bool = False
    use_repair_context: bool = False
    context_operation: Literal["KEEP", "CORRECT", "PAUSE", "CANCEL", "RESUME", "REPLACE"] = "KEEP"
    action_request: Literal["NONE", "INFORMATION_ONLY", "EXPLICIT", "CONDITIONAL"] = "NONE"
    repair_request_quote: str | None = Field(default=None, max_length=200)
    condition_quote: str | None = Field(default=None, max_length=200)
    withdraw_condition_quote: str | None = Field(default=None, max_length=200)
    missing_slot: Literal["NONE", "location", "part", "symptom", "available_time", "correction_target", "order_number", "condition_choice"] = "NONE"
    inline_reply: ReplyDraft | None = None
    boundary_quote: str | None = Field(default=None, max_length=200)


def strict_schema(model=DialogueDecision):
    schema = model.model_json_schema()

    def visit(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            node.pop("default", None)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    visit(schema)
    return schema


def compact_schema(model=DialogueDecision):
    """Omit default wire fields; runtime validation still expands and checks the full model."""
    schema = model.model_json_schema()

    def visit(node):
        if isinstance(node, dict):
            node.pop("default", None)
            node.pop("title", None)
            if "segments" in node.get("properties", {}):
                node["required"] = ["segments", "addressed_tasks", "question_slot"]
            if 'additional_fact_required' in node.get('properties', {}):
                # False differs from the conservative server default True.
                # Omitting it must never silently reject a full semantic match.
                node['required'] = list(dict.fromkeys([*node.get('required', []),
                    'additional_fact_required', 'question_relation']))
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    visit(schema)
    return schema
