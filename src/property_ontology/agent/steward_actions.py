"""Execute semantic tasks against controlled Actions; retain separate outcomes and conditions."""

import copy
import re

from ..domain import POCError, canonical
from ..qa import AnswerResult
from ..response_policy import fixed_reply
from .planner import ORDER_NUMBER


def execute_steward_actions(session, decision, message, request_key):
    from .dialogue import (
        FREE_CONDITION,
        RELEASE_CONDITION,
        authorizes_semantic_repair,
        decision_context,
        validate_decision,
    )
    validate_decision(decision, message, decision_context(session))
    # A model's EXPLICIT label is not user consent. Downgrade an unsupported
    # prepare suggestion to discussion before any draft or confirmation is made.
    prior_task = session.conversation_state.get("active_task") or {}
    slot_updates = decision.repair.model_dump(exclude_none=True)
    changed_slots = any(session.repair_details.get(k) != v for k, v in slot_updates.items())
    continuation = bool(prior_task.get("user_requested_prepare") and changed_slots)
    if (any(t.intent == "REPAIR_PREPARE" for t in decision.tasks)
            and not authorizes_semantic_repair(decision, message)
            and not continuation and not decision.condition_quote and not decision.requires_free):
        decision = decision.model_copy(update={"action_request": "INFORMATION_ONLY", "tasks": [
            t.model_copy(update={"intent": "REPAIR_GUIDANCE"}) if t.intent == "REPAIR_PREPARE" else t
            for t in decision.tasks]})
    fallback_lookup = False
    original_route = decision.route
    knowledge_correction = (decision.route == 'CORRECTION'
        and session.conversation_state.get('current_topic') == 'KNOWLEDGE'
        and session.conversation_state.get('knowledge_context')
        and not decision.repair.model_dump(exclude_none=True)
        and decision.action_request in {'NONE', 'INFORMATION_ONLY'})
    if knowledge_correction:
        from .dialogue_contract import DialogueTask
        # Correcting an answer must not cancel a repair draft or skip retrieval.
        # The semantic matcher sees the previous question and this correction.
        decision = decision.model_copy(update={'route':'SERVICE', 'context_operation':'KEEP',
            'tasks':[DialogueTask(intent='KNOWLEDGE', source_quote=message, query=message)]})
    if hasattr(session.planner, "match_templates") and (decision.route == "OUT_OF_SCOPE"
            or decision.tasks and all(t.intent == "EXPLAIN_LIMITS" for t in decision.tasks)):
        # A model's belief that knowledge is unavailable is not a search result.
        # Only a read-only, estate-scoped lookup can establish that limitation.
        from .dialogue_contract import DialogueTask
        decision = decision.model_copy(update={"route": "SERVICE", "tasks": [DialogueTask(
            intent="KNOWLEDGE", source_quote=message, query=message)]})
        fallback_lookup = True
    state = session.conversation_state
    state["request_key"] = request_key or state.get("request_key", "")
    intents = {t.intent for t in decision.tasks}
    state["current_intent"] = ",".join(t.intent for t in decision.tasks) if decision.route == "SERVICE" else decision.route
    if decision.route == "OUT_OF_SCOPE" or intents == {"EXPLAIN_LIMITS"}:
        state["current_topic"] = "EXTERNAL"
    elif intents == {"SOCIAL"}:
        state["current_topic"] = "SOCIAL"
    elif any(i.startswith("REPAIR_") or i in {"TRACK_ORDER", "OPERATION_STATUS"} for i in intents) or decision.route in {"CANCEL", "CORRECTION"}:
        state["current_topic"] = "REPAIR"
    elif "KNOWLEDGE" in intents:
        state["current_topic"] = "KNOWLEDGE"
    state["error_or_correction_state"] = {'kind':'KNOWLEDGE_CORRECTION'} if knowledge_correction else None
    outcomes, texts, knowledge = [], [], []
    prepared = False

    def outcome(intent, status, **extra):
        item = {"task_index": len(outcomes), "intent": intent, "status": status, **extra}
        outcomes.append(item)
        return item

    def finish(text, kind="DIALOGUE"):
        state["pending_tasks"] = [{"intent": o["intent"], "status": o["status"]} for o in outcomes]
        state["collected_slots"] = copy.deepcopy(session.repair_details)
        if knowledge:
            if len(knowledge) == 1:
                answer = copy.deepcopy(knowledge[0])
                answer["answer_text"] = text
                answer["debug_info"].update(dialogue_outcomes=outcomes, decision=decision.model_dump())
                return answer
            sources = {canonical(e): e for a in knowledge for e in a["evidence_refs"]}
            warnings = [w for a in knowledge for w in a["warnings"]]
            status = "ANSWERED" if all(a["status"] == "ANSWERED" for a in knowledge) else next(a["status"] for a in knowledge if a["status"] != "ANSWERED")
            return AnswerResult(status=status, answer_text=text, estate_id=session.estate_id,
                answer_source="TEMPLATE" if all(a.get("answer_source") == "TEMPLATE" for a in knowledge) else "KNOWLEDGE_GENERATION",
                response_mode="TEMPLATE_MULTI" if all(a.get("answer_source") == "TEMPLATE" for a in knowledge) else "KNOWLEDGE_GENERATION",
                quote_notice=next((a['quote_notice'] for a in knowledge if a.get('quote_notice')), None),
                evidence_refs=list(sources.values()), matched_policy_ids=list(dict.fromkeys(i for a in knowledge for i in a["matched_policy_ids"])),
                warnings=list(dict.fromkeys(warnings)), debug_info={"dialogue_outcomes": outcomes,
                "subanswers": [{"answer": a} for a in knowledge], "decision": decision.model_dump()}).model_dump()
        return {"kind": kind, "message": text, "dialogue_outcomes": outcomes}

    if decision.route in {"RED_LINE", "HUMAN_REQUEST"}:
        return fixed_reply(session.estate_id, "RED_LINE" if decision.route == "RED_LINE" else "USER_REQUEST")
    if decision.context_operation == "REPLACE":
        session._cancel_pending()
        if state["active_task"]:
            state["pending_tasks"] = [*state["pending_tasks"][-3:], {**state["active_task"], "status": "PAUSED"}]
        session.repair_dialogue = []
        session.repair_details.clear()
    elif decision.context_operation == "PAUSE":
        session._cancel_pending()
        state["active_task"] = {"kind": "repair", "status": "PAUSED"}

    # Slot values have already been checked against original user messages, not assistant output.
    updates = decision.repair.model_dump(exclude_none=True)
    had_pending = bool(session.pending)
    previous_authority = had_pending or bool(state["active_task"] and state["active_task"].get("user_requested_prepare"))
    changed_updates = any(session.repair_details.get(k) != v for k, v in updates.items())
    changed_pending = had_pending and changed_updates
    if changed_pending:
        session._cancel_pending()
    session.repair_details.update(updates)
    state["collected_slots"] = copy.deepcopy(session.repair_details)
    if updates and message not in session.repair_dialogue:
        session.repair_dialogue = [*session.repair_dialogue[-2:], message]

    condition = decision.condition_quote
    if FREE_CONDITION.search(message):
        condition = message
    if condition:
        session._cancel_pending()
        kind = "FREE" if decision.requires_free or FREE_CONDITION.search(condition) else "USER_CONDITION"
        if not any(c["text"] == condition and c["status"] == "UNRESOLVED" for c in state["pending_conditions"]):
            state["pending_conditions"].append({"kind": kind, "text": condition, "status": "UNRESOLVED"})
    if decision.requires_free and not any(c["kind"] == "FREE" and c["status"] == "UNRESOLVED" for c in state["pending_conditions"]):
        state["pending_conditions"].append({"kind": "FREE", "text": message, "status": "UNRESOLVED"})
    release = bool(RELEASE_CONDITION.search(message))
    withdrawal = decision.withdraw_condition_quote
    if withdrawal and withdrawal not in message:
        raise POCError("INVALID_TOOL_INPUT", "撤回条件需要引用您本轮的原话。", 400)
    explicit_withdrawal = withdrawal and bool(re.search(r"不再要求|取消.{0,6}条件|不加条件|无论|不管|不用.{0,6}限制", withdrawal))
    if release or explicit_withdrawal:
        for c in state["pending_conditions"]:
            if explicit_withdrawal or c["kind"] == "FREE":
                c["status"] = "WITHDRAWN"
        session.repair_requires_free = False
    elif decision.requires_free or FREE_CONDITION.search(message):
        session.repair_requires_free = True
    if session.repair_requires_free and not any(c["kind"] == "FREE" and c["status"] == "UNRESOLVED" for c in state["pending_conditions"]):
        state["pending_conditions"].append({"kind": "FREE", "text": "此前提出的免费前提", "status": "UNRESOLVED"})
    unresolved = [c for c in state["pending_conditions"] if c["status"] == "UNRESOLVED"]
    state["pending_conditions"] = state["pending_conditions"][-8:]

    if decision.route in {"CORRECTION", "CANCEL"} or decision.context_operation in {"CORRECT", "CANCEL"}:
        session._cancel_pending()
        route = "CANCEL" if decision.route == "CANCEL" or decision.context_operation == "CANCEL" else "CORRECTION"
        state["active_task"] = {"kind": "repair", "status": "PAUSED" if route == "CANCEL" else "NEEDS_CORRECTION"}
        state["error_or_correction_state"] = {"kind": route, "updated_fields": list(updates), "submitted_order": session.last_order}
        state["missing_slots"] = [] if updates or route == "CANCEL" else ["correction_target"]
        outcome(route, "SUBMITTED_ORDER_UNCHANGED" if session.last_order else "DRAFT_STOPPED", updated_fields=list(updates))
        text = ("这次未提交的报修先停下，之前说的故障和时间还记着。" if route == "CANCEL" else
                "我先停下旧的待确认内容，已知信息会保留。" + ("请告诉我哪项需要更正。" if not updates else "你指出的信息会用于重新核对。"))
        if session.last_order:
            text = "原工单已经提交，尚未撤销或修改；需要核对你希望更正的内容，当前不会另建一张。"
        if route == "CANCEL" or not any(t.intent == "REPAIR_PREPARE" for t in decision.tasks):
            return finish(text, "CANCELLED" if route == "CANCEL" else "CORRECTED")
        texts.append(text)
    if decision.route == "OUT_OF_SCOPE":
        outcome("EXPLAIN_LIMITS", "NO_RELIABLE_EXTERNAL_DATA")
        return finish("我还没有可靠的附近商家信息，暂时没法给出具体推荐。")
    if decision.route == "CLARIFY":
        state["missing_slots"] = [decision.missing_slot] if decision.missing_slot != "NONE" else []
        outcome("CLARIFY", "NEEDS_CONTEXT", missing_slot=decision.missing_slot)
        return finish("刚才的意思我还没理解准确。请补充你希望我继续做的那一点。", "CLARIFY")

    state["missing_slots"] = []
    continuing_collection = bool(state["active_task"] and state["active_task"].get("status") == "NEEDS_INFORMATION" and updates)
    prepare_task = decision.tasks[0] if (changed_pending or continuing_collection) and decision.route == "SERVICE" and decision.tasks else None
    for task in decision.tasks:
        if task.intent == "KNOWLEDGE":
            query = task.query or task.source_quote
            if session.foreign_scope(query):
                raise POCError("INVALID_TOOL_INPUT", "当前只能处理已登录楼盘的资料。", 400)
            try:
                semantic = {"semantic_matcher": session.planner, "semantic_context": {
                    "user_question": task.source_quote, "resolved_query": query,
                    "conversation_history": state["conversation_history"][-8:],
                    "knowledge_context": state.get("knowledge_context"),
                    "_planning_diagnostics": {k: v for k, v in getattr(session.planner, 'last_metrics', {}).items()
                                               if k.startswith('joint_knowledge_')},
                    "_joint_selections": getattr(session, '_knowledge_matches', [])}} if hasattr(session.planner, "match_templates") else {}
                answer = session.actions.answer_question(session.token, query, service_code=None if task.service_code == "NONE" else task.service_code,
                    unit_id=session.unit_id, knowledge_mode=session.knowledge_mode, renovation_activity=session.renovation_activity, **semantic)
                if fallback_lookup:
                    answer["debug_info"]["pre_fallback_lookup"] = {"original_route": original_route, "performed": True}
            except POCError as exc:
                answer = AnswerResult(status="NOT_VERIFIED", answer_text="这项信息暂时没有核实，稍后可以再查。",
                    estate_id=session.estate_id, answer_source="CLARIFICATION", warnings=[exc.code],
                    debug_info={"tool_error": exc.code, "diagnostics": getattr(exc, "diagnostics", None)}).model_dump()
            knowledge.append(answer)
            state["knowledge_context"] = {"question": task.source_quote, "resolved_query": query, "service_code": task.service_code,
                                           "matched_template_id": answer.get("matched_template_id")}
            outcome(task.intent, answer["status"], query=query, answer_text=answer["answer_text"] if answer["status"] == "ANSWERED" else None,
                    answer_source=answer.get("answer_source"), coverage=answer.get("debug_info", {}).get("generation_sufficiency"),
                    source_type=answer.get("source_type"), reason=answer.get("decision_reason"))
            outcomes[-1]['quote_notice'] = answer.get('quote_notice')
            texts.append(answer["answer_text"])
        elif task.intent in {"REPAIR_GUIDANCE", "REPAIR_CAUSE"}:
            if not session.pending:
                state["active_task"] = {"kind": "repair", "status": "DISCUSSING", "user_requested_prepare": previous_authority}
            if message not in session.repair_dialogue:
                previous = session.repair_dialogue
                session.repair_dialogue = [*previous, message] if len(previous) < 3 else [previous[0], previous[-1], message]
            missing = [] if task.intent == "REPAIR_CAUSE" or session.repair_details.get("part") or "电梯" in "\n".join(session.repair_dialogue) else ["part"]
            state["missing_slots"] = missing
            outcome(task.intent, "CAUSE_UNVERIFIED", known_slots=dict(session.repair_details), missing_slots=missing)
            if "电梯" in "\n".join(session.repair_dialogue):
                texts.append("请先别使用故障电梯。能否今天修好，需要检修人员现场确认；本次尚未提交新的报修。")
            else:
                texts.append("仅凭描述还不能确定漏水原因，需要现场检查。若能安全操作，可先关闭供水阀门，远离积水附近的插座和电器。")
                if missing:
                    texts.append("能看出水主要从哪里渗出吗？")
        elif task.intent == "REPAIR_PREPARE":
            prepare_task = task
        elif task.intent == "OPERATION_STATUS":
            from .operation_state import operation_answer, operation_snapshot
            snapshot = operation_snapshot(session)
            text = operation_answer(snapshot, task.operation_subject)
            outcome(task.intent, "ANSWERED", answer_text=text, operation_state=snapshot)
            texts.append(text)
        elif task.intent == "TRACK_ORDER":
            number = ORDER_NUMBER.search(message)
            number = number.group(0) if number else session.last_order
            if not number:
                state["missing_slots"] = ["order_number"]
                outcome(task.intent, "NEED_ORDER_NUMBER")
                texts.append("把本人工单编号发我，就能帮你查进度。")
            else:
                from .presentation import present
                try:
                    result = session.actions.get_work_order_status(session.token, number)
                    text = present(result)["text"]
                    state["last_tool_result"] = copy.deepcopy(result)
                    outcome(task.intent, "ANSWERED", answer_text=text)
                    texts.append(text)
                except POCError as exc:
                    if exc.status == 401:
                        raise
                    outcome(task.intent, "NOT_AVAILABLE", error_code=exc.code)
                    texts.append("这张工单目前无法查询，可能不存在或当前账号无权查看。")
        elif task.intent == "COMPLAINT":
            outcome(task.intent, "ACTION_NOT_IMPLEMENTED")
            texts.append("我理解你希望投诉，目前还不能在这里创建投诉工单；可以先查询投诉渠道。")
        else:
            outcome(task.intent, "CONVERSATION", query=task.query or task.source_quote)
            texts.append("刚才没说清楚的地方，我们可以接着说。" if task.intent == "FEEDBACK" else "你好，我在。")

    if unresolved and (prepare_task or FREE_CONDITION.search(message)):
        session._cancel_pending()
        state["active_task"] = {"kind": "repair", "status": "WAITING_CONDITIONS"}
        state["missing_slots"] = ["condition_choice"]
        outcome("REPAIR_PREPARE", "CONDITION_UNRESOLVED", conditions=unresolved)
        texts.append("目前无法确认这次维修是否免费或是否满足你提出的条件，因此先不提交报修。")
        prepare_task = None
    if prepare_task:
        authorized = authorizes_semantic_repair(decision, message) or (previous_authority and changed_updates)
        if decision.action_request == "INFORMATION_ONLY":
            authorized = False
        if not authorized:
            outcome("REPAIR_PREPARE", "NEED_USER_INTENT")
            texts.append("要不要把这次故障准备成报修？你确认内容后才会提交。")
        elif not session.repair_details.get("symptom"):
            state["active_task"] = {"kind": "repair", "status": "NEEDS_INFORMATION", "user_requested_prepare": True}
            state["missing_slots"] = ["symptom"]
            outcome("REPAIR_PREPARE", "MISSING_FAULT_DESCRIPTION", missing_slots=["symptom"])
            texts.append("哪里出了什么问题？说一下故障现象，我再准备报修内容。")
        elif session.pending_requires_recheck and not (condition or release or explicit_withdrawal):
            outcome("REPAIR_PREPARE", "SUPPLEMENT_UNVERIFIED")
            texts.append("刚才补充的要求还没核对清楚，请重新说明需要满足的条件后再准备。")
        else:
            session._cancel_pending()
            if not session.unit_id:
                raise POCError("UNIT_REQUIRED", "先选择本人关联的房屋，才能准备报修。", 400)
            if message not in session.repair_dialogue:
                old = session.repair_dialogue
                session.repair_dialogue = [*old, message] if len(old) < 3 else [old[0], old[-1], message]
            labels = {"location": "位置", "part": "部位", "symptom": "现象", "started_at": "发现时间", "available_time": "方便上门的时间意向"}
            raw = "\n".join(labels[field] + "：" + value for field, value in session.repair_details.items() if value)
            if len(raw) > 1000:
                raise POCError("INVALID_INPUT", "故障信息较长，请简化后重新准备。", 400)
            result = session.actions.prepare_work_order(session.token, {"unit_id": session.unit_id, "description": raw,
                "service_code": "maintenance", "idempotency_key": request_key})
            outcome("REPAIR_PREPARE", "PREPARED_NOT_SUBMITTED", known_slots=dict(session.repair_details))
            state["active_task"] = {"kind": "repair", "status": "WAITING_CONFIRMATION", "user_requested_prepare": True}
            state["last_tool_result"] = {"operation": "prepare_repair", "status": "PREPARED_NOT_SUBMITTED", "created": False}
            state["missing_slots"] = []
            state["pending_tasks"] = [{"intent": o["intent"], "status": o["status"]} for o in outcomes]
            texts.append("报修内容准备好了。请核对卡片，点击确认后才会提交；上门时间仍需维修人员确认。")
            result.update(assistant_text="\n".join(dict.fromkeys(texts)), dialogue_details=dict(session.repair_details),
                          knowledge_results=knowledge, dialogue_outcomes=outcomes)
            prepared = True
            return result
    if not prepared:
        return finish("\n".join(dict.fromkeys(texts)) or "我在听，你可以接着说。")
