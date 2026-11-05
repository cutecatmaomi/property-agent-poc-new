"""Stable conversation graph; immutable approval snapshots remain in separate child threads."""

import copy
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ..domain import POCError
from ..response_policy import classify_message
from .dialogue_contract import DialogueDecision

HISTORY_LIMIT = 24


class ConversationState(TypedDict, total=False):
    session_id: str
    thread_id: str
    authenticated_resident_id: str
    estate_id: str
    current_user_message: str
    conversation_history: list[dict]
    current_intent: str | None
    current_topic: str | None
    active_task: dict | None
    pending_tasks: list[dict]
    collected_slots: dict
    missing_slots: list[str]
    pending_confirmation: dict | None
    pending_conditions: list[dict]
    last_tool_result: dict | None
    knowledge_evidence: list[dict]
    knowledge_context: dict | None
    error_or_correction_state: dict | None
    final_answer: str
    request_key: str
    reply: dict
    skip_execution: bool


def new_state(session_id, resident, estate):
    return ConversationState(session_id=session_id, thread_id=session_id, authenticated_resident_id=resident,
        estate_id=estate, current_user_message="", conversation_history=[], current_intent=None, current_topic=None, active_task=None,
        pending_tasks=[], collected_slots={}, missing_slots=[], pending_confirmation=None, pending_conditions=[],
        last_tool_result=None, knowledge_evidence=[], knowledge_context=None, error_or_correction_state=None, final_answer="")


def remember(session, message, reply, *, event=False):
    history = session.conversation_state["conversation_history"]
    topic = session.conversation_state["current_topic"]
    quote = reply.get('data', {}).get('quote_notice')
    remembered_answer = (quote + '\n' if quote else '') + reply['text']
    history.extend([{"role": "user", "content": message, "event": event, "topic": topic},
                    {"role": "assistant", "content": remembered_answer, "kind": reply["kind"], "topic": topic}])
    session.conversation_state["conversation_history"] = history[-HISTORY_LIMIT:]
    session.conversation_state["current_user_message"] = message
    session.conversation_state["final_answer"] = reply["text"]
    session.conversation_state["collected_slots"] = copy.deepcopy(session.repair_details)
    session.conversation_state["pending_confirmation"] = copy.deepcopy(session.pending)
    session.last_reply_kind = reply["kind"]
    session.dialogue_history = [m["content"] for m in history if m["role"] == "user"][-12:]
    data = reply.get("data", {})
    if data.get("evidence_refs"):
        session.conversation_state["knowledge_evidence"] = copy.deepcopy(data["evidence_refs"])
    if reply["kind"] == "ORDER_CREATED":
        session.conversation_state["active_task"] = {"kind": "repair", "status": "SUBMITTED", "number": data["number"]}
        session.conversation_state["last_tool_result"] = {"number": data["number"], "status": data["status"], "replayed": data.get("replayed", False)}
    elif reply["kind"] == "ORDER_STATUS":
        session.conversation_state["last_tool_result"] = copy.deepcopy(data)
    elif reply["kind"] == "REPAIR_PREVIEW":
        session.conversation_state["active_task"] = {"kind": "repair", "status": "WAITING_CONFIRMATION"}


def build_conversation_graph(session, checkpointer):
    def understand(state):
        session._planned_decision = None
        session._inline_decision = None
        session._knowledge_matches = []
        message = state["current_user_message"]
        try:
            session.verify()
            from ..qa.day1_poc import exact_question
            if exact_question(session, message) and not classify_message(message) and not session.foreign_scope(message):
                # Exact approved questions are read-only information requests. The
                # regular tool path still checks login, scope, mode and source version.
                decision = DialogueDecision.model_validate({'route': 'SERVICE', 'action_request': 'INFORMATION_ONLY',
                    'tasks': [{'intent': 'KNOWLEDGE', 'source_quote': message, 'query': message}]})
                session._planned_decision = session._inline_decision = decision
                return {'skip_execution': False}
            if hasattr(session.planner, "decide") and not classify_message(message) and not session.foreign_scope(message):
                from .dialogue import decision_context, validate_decision
                context = decision_context(session)
                if (getattr(session.planner, 'joint_knowledge_available', False)
                        and getattr(session.planner.settings, 'joint_knowledge_planning', False)
                        and session.knowledge_mode == 'demo'):
                    from ..qa.day1_poc import load, projected, session_root
                    root = session_root(session)
                    packet = load(root) if root else None
                    if packet:
                        rows, receipt, _ = packet
                        context.update(knowledge_set=packet[2]['active_set'],
                            question_set_id=packet[2].get('question_set_id'))
                        context['_template_catalog'] = [projected(r, receipt) for r in rows[session.estate_id]]
                try:
                    planned = session.planner.decide(message, context=context)
                except POCError as exc:
                    if exc.code != "INVALID_TOOL_PLAN":
                        raise
                    # A malformed plan has executed nothing. One protocol repair is
                    # safe; auth/network failures are never retried automatically.
                    planned = session.planner.decide(message, context={**context,
                        "protocol_retry": {"instruction": "修复格式。route非SERVICE时tasks必须为空；任务intent不能填route枚举。NONE字段不要用null。不要从历史重新发起业务。",
                                           "errors": getattr(session.planner, "last_plan_error", None)}})
                decision = DialogueDecision.model_validate(planned)
                validate_decision(decision, message, context)
                session._knowledge_matches = getattr(planned, '_knowledge_matches', [])
                session._planned_decision = decision
                session._inline_decision = decision
                session.conversation_state["current_intent"] = decision.route
            return {"skip_execution": False}
        except POCError as exc:
            if session._pending and session._turns[session._pending]["accepted"] is None:
                session.pending_requires_recheck = True
            session._turns.pop(state["request_key"], None)
            session.conversation_state["error_or_correction_state"] = {"kind": "MODEL_ERROR", "code": exc.code}
            from .presentation import error_reply
            return {"skip_execution": True, "reply": error_reply(exc)}

    def execute(state):
        key = state["request_key"]
        try:
            result = session._graph.invoke({"message": state["current_user_message"], "request_key": key, "trace": []}, session._config(key))
            return {"reply": session._finish(key, result)}
        except POCError as exc:
            if session._pending and session._turns[session._pending]["accepted"] is None:
                session.pending_requires_recheck = True
            session._memory.delete_thread(key)
            session._turns.pop(key, None)
            session.conversation_state["error_or_correction_state"] = {"kind": "TOOL_ERROR", "code": exc.code}
            from .presentation import error_reply
            return {"reply": error_reply(exc)}
        finally:
            session._planned_decision = None
            session._knowledge_matches = []

    def compose(state):
        from .reply_generation import generate_reply
        try:
            return {"reply": generate_reply(session, state["current_user_message"], state["reply"])}
        finally:
            session._inline_decision = None

    def retain(state):
        remember(session, state["current_user_message"], state["reply"])
        return {**copy.deepcopy(session.conversation_state), "reply": state["reply"]}

    graph = StateGraph(ConversationState)
    for name, fn in (("understand", understand), ("execute", execute), ("compose", compose), ("template_direct", compose), ("remember", retain)):
        graph.add_node(name, fn)
    graph.add_edge(START, "understand")
    graph.add_conditional_edges("understand", lambda s: "remember" if s["skip_execution"] else "execute")
    graph.add_conditional_edges("execute", lambda s: "template_direct" if
        s['reply'].get('data', {}).get('answer_source') == 'TEMPLATE' and
        all(o.get('intent') == 'KNOWLEDGE' for o in s['reply'].get('data', {}).get('debug_info', {}).get('dialogue_outcomes', [])) else "compose")
    graph.add_edge("template_direct", "remember")
    graph.add_edge("compose", "remember")
    graph.add_edge("remember", END)
    return graph.compile(checkpointer=checkpointer)
