"""Real LangGraph orchestration; only a trusted UI operation can resume the approval interrupt."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import ValidationError

from ..domain import POCError, canonical
from ..response_policy import classify_message, fixed_reply
from .dialogue import decision_context, execute_dialogue, validate_decision
from .dialogue_contract import DialogueDecision
from .planner import CLARIFICATIONS, PLAN, TOOL_NAMES
from .presentation import present


class TurnState(TypedDict, total=False):
    message: str
    request_key: str
    trace: list[str]
    route: str
    plan: dict
    result: dict
    preview: dict
    approved: bool
    reply: dict


def build_graph(session, checkpointer):
    def step(state, name, **update):
        return {"trace": [*state.get("trace", []), name], **update}

    def guard(state):
        session.verify()
        reason = classify_message(state["message"])
        if reason:
            return step(state, "guard", route="render", result=fixed_reply(session.estate_id, reason))
        if session.foreign_scope(state["message"]):
            return step(state, "guard", route="render", result={"kind": "CLARIFY", "message":
                        "当前仅可查询或办理已登录楼盘的业务；如需其他楼盘，请退出后选择对应楼盘登录。"})
        return step(state, "guard", route="plan")

    def plan(state):
        try:
            if hasattr(session.planner, "decide") or getattr(session, '_planned_decision', None) is not None:
                context = decision_context(session)
                decision = getattr(session, "_planned_decision", None)
                if decision is None:
                    decision = DialogueDecision.model_validate(session.planner.decide(state["message"], context=context))
                validate_decision(decision, state["message"], context)
                session.actions.auth.audit(session.estate_id, session._binding[1], "agent_semantic_plan",
                                           canonical({"decision": decision.model_dump(),
                                                      "model_metrics": getattr(session.planner, "last_metrics", {})}),
                                           state["request_key"])
                return step(state, "plan", plan=decision.model_dump(), route="dialogue")
            call = PLAN.validate_python(session.planner.plan(state["message"], last_order=session.last_order))
        except (ValidationError, ValueError, TypeError, RuntimeError) as exc:
            raise POCError("INVALID_TOOL_PLAN", "暂时无法识别该请求，请换一种说法；没有执行任何业务操作", 400) from exc
        return step(state, "plan", plan=call.model_dump(), route=call.name)

    def dialogue(state):
        session.verify()
        result = execute_dialogue(session, DialogueDecision.model_validate(state["plan"]), state["message"], state["request_key"])
        if result.get("requires_user_confirmation"):
            return step(state, "dialogue", result=result, preview=result, route="approval")
        return step(state, "dialogue", result=result, route="render")

    def tool(state):
        session.verify()
        result = session.tools.execute(state["plan"], message=state["message"], unit_id=session.unit_id,
                                       knowledge_mode=session.knowledge_mode,
                                       renovation_activity=session.renovation_activity,
                                       request_key=state["request_key"])
        if result.get("requires_user_confirmation"):
            return step(state, state["plan"]["name"], result=result, preview=result, route="approval")
        return step(state, state["plan"]["name"], result=result, route="render")

    def clarify(state):
        # Even a replacement planner cannot inject unreviewed policy prose through clarification.
        text = CLARIFICATIONS[state["plan"]["arguments"]["code"]]
        return step(state, "clarify", result={"kind": "CLARIFY", "message": text})

    def approval(state):
        preview = state["preview"]
        answer = interrupt({"kind": "REPAIR_CONFIRMATION", "unit_id": preview["unit_id"],
                            "description": preview["description"], "expires_in_seconds": 600,
                            "assistant_text": preview.get("assistant_text"),
                            "dialogue_details": preview.get("dialogue_details", {}),
                            "dialogue_outcomes": preview.get("dialogue_outcomes", []),
                            "knowledge_results": preview.get("knowledge_results", [])})
        session.verify()
        if not session.valid_resume(state["request_key"], answer):
            raise POCError("CONFIRMATION_REQUIRED", "请使用当前报修卡片的确认按钮", 409)
        accepted = answer["accept"]
        session.actions.confirm_work_order(session.token, preview["confirmation_ref"], accept=accepted)
        return step(state, "ui_confirm", approved=accepted, result={"kind": "CANCELLED", "message": "已取消本次报修，未创建工单。"})

    def create(state):
        session.verify()
        p = state["preview"]
        result = session.actions.create_work_order(session.token, {
            "unit_id": p["unit_id"], "description": p["description"], "service_code": "maintenance",
            "idempotency_key": p["idempotency_key"], "confirmation_ref": p["confirmation_ref"],
        })
        return step(state, "create_work_order", result=result)

    def render(state):
        return step(state, "render", reply=present(state["result"]))

    graph = StateGraph(TurnState)
    for name, fn in {"guard": guard, "plan": plan, "dialogue": dialogue, "consult": tool, "prepare_repair": tool,
                     "track_order": tool, "clarify": clarify, "approval": approval,
                     "create": create, "render": render}.items():
        graph.add_node(name, fn)
    graph.add_edge(START, "guard")
    graph.add_conditional_edges("guard", lambda s: s["route"], {"plan": "plan", "render": "render"})
    graph.add_conditional_edges("plan", lambda s: s["route"], {n: n for n in (*TOOL_NAMES, "clarify", "dialogue")})
    for name in (*TOOL_NAMES, "dialogue"):
        graph.add_conditional_edges(name, lambda s: s["route"], {"approval": "approval", "render": "render"})
    graph.add_edge("clarify", "render")
    graph.add_conditional_edges("approval", lambda s: "create" if s["approved"] else "render")
    graph.add_edge("create", "render")
    graph.add_edge("render", END)
    return graph.compile(checkpointer=checkpointer)


