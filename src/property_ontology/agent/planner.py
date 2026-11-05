"""Replaceable structured planner. No model API, arbitrary tools, identities or business approval."""

import re
from typing import Annotated, Literal, Protocol

from pydantic import Field, TypeAdapter

from ..domain import Strict
from ..response_policy import normalized_text

Service = Literal["activity", "membership", "complaint", "renovation", "property_fee", "parking", "warranty",
                  "maintenance", "delivery", "purchase", "city_info", "property_service", "project_info"]


class ConsultArgs(Strict):
    question: str = Field(min_length=1, max_length=300)
    service_code: Service | None = None


class RepairArgs(Strict):
    description: str = Field(min_length=1, max_length=300)


class TrackArgs(Strict):
    number: str = Field(min_length=6, max_length=160)


class ClarifyArgs(Strict):
    code: Literal["CONFIRM_BUTTON", "ORDER_NUMBER_REQUIRED", "REPAIR_DESCRIPTION_REQUIRED", "REPAIR_OR_RULE"]


class Consult(Strict):
    name: Literal["consult"]
    arguments: ConsultArgs


class Repair(Strict):
    name: Literal["prepare_repair"]
    arguments: RepairArgs


class Track(Strict):
    name: Literal["track_order"]
    arguments: TrackArgs


class Clarify(Strict):
    name: Literal["clarify"]
    arguments: ClarifyArgs


PLAN = TypeAdapter(Annotated[Consult | Repair | Track | Clarify, Field(discriminator="name")])
TOOL_NAMES = ("consult", "prepare_repair", "track_order")
ORDER_NUMBER = re.compile(r"(?<![A-Za-z0-9])E00[12]-(?:MOCK-)?[A-Za-z0-9-]{8,100}(?![A-Za-z0-9-])", re.IGNORECASE)
CLARIFICATIONS = {
    "CONFIRM_BUTTON": "请核对报修卡片，并点击“确认提交报修”按钮；聊天文字不会确认或创建工单。",
    "ORDER_NUMBER_REQUIRED": "请提供本人工单编号，例如复制报修成功后显示的完整编号。",
    "REPAIR_DESCRIPTION_REQUIRED": "请说明具体故障，例如“我要报修，厨房水龙头滴水”，并在侧栏选择本人的房屋。",
    "REPAIR_OR_RULE": "如需提交报修，请描述具体故障；如需了解规则，可以问“报修流程是什么”。",
}


class Planner(Protocol):
    def plan(self, message: str, *, last_order: str | None) -> dict: ...


class LocalDemoPlanner:
    """Deterministic stand-in for the future LLM's structured tool selection, not a real LLM."""

    def plan(self, message, *, last_order=None):
        text = normalized_text(message)
        if text in {"确认", "确认提交", "提交", "yes", "confirm"}:
            return self.clarify("CONFIRM_BUTTON")
        number = ORDER_NUMBER.search(message)
        explicit = any(w in text for w in ("我要报修", "帮我报修", "申请报修", "提交报修", "报修一下"))
        tracking = bool(number) or any(w in text for w in ("进度", "状态", "追踪", "跟踪")) or (
            "工单" in text and any(w in text for w in ("查", "查看")) and not any(w in text for w in ("流程", "规则", "如何", "怎么")))
        if tracking and not explicit:
            if number or last_order:
                return {"name": "track_order", "arguments": {"number": number.group(0) if number else last_order}}
            return self.clarify("ORDER_NUMBER_REQUIRED")
        symptom = any(w in text for w in ("漏水", "滴水", "堵塞", "不亮", "坏了", "没电", "故障", "打不开"))
        asks_rule = any(w in text for w in ("如何", "怎么", "多少", "多久", "什么时候", "流程", "收费", "保修", "是否"))
        if explicit or symptom and not asks_rule:
            if text in {"我要报修", "帮我报修", "申请报修", "提交报修", "报修一下"}:
                return self.clarify("REPAIR_DESCRIPTION_REQUIRED")
            return {"name": "prepare_repair", "arguments": {"description": message}}
        if text in {"报修", "维修"}:
            return self.clarify("REPAIR_OR_RULE")
        return {"name": "consult", "arguments": {"question": message}}

    @staticmethod
    def clarify(code):
        return {"name": "clarify", "arguments": {"code": code}}
