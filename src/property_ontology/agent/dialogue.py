"""Conversation execution policy: semantic plans are suggestions; facts/actions stay server controlled."""

import re

from ..domain import POCError
from .dialogue_contract import DialogueDecision, DialogueTask, RepairSlots
from .planner import LocalDemoPlanner

FREE_CONDITION = re.compile(r"(?:如果|只要|仅|只有|要是|得|必须).{0,10}(?:免费|不收费)|(?:免费|不收费).{0,8}(?:才|就|再).{0,8}(?:报修|安排|维修|提交)")
RELEASE_CONDITION = re.compile(r"(?:收费|付费).{0,8}(?:也可以|也行|也报修|也修|接受)|不管.{0,8}(?:收费|免费)|不要求免费")
CORRECTION = re.compile(r"(?:创建|建单|报修|弄|搞|选|理解|填).{0,3}错|不是.{0,6}(?:投诉|报修)|你.{0,3}误会")
CANCEL = re.compile(r"取消|先别.{0,4}(?:报修|提交|建单)|不要.{0,4}(?:报修|提交|建单)")
EXPLICIT_REPAIR = re.compile(r"(?:帮|请|我要|我需要|麻烦|安排|提交|申请|直接).{0,12}(?:报(?:个|一下)?修|维修|师傅|上门|修一下|处理|人.{0,3}看)|来个.{0,3}(?:人|师傅)|^报修$")


def authorizes_repair(message):
    if re.search(r"(?:怎么|如何|怎样).{0,5}报(?:个|一下)?修|报修.{0,4}(?:方法|渠道|流程)", message) and not re.search(r"(?:帮我|我要|请|申请|提交|直接).{0,4}报(?:个|一下)?修", message):
        return False
    return bool(EXPLICIT_REPAIR.search(message))


def authorizes_semantic_repair(decision, message):
    """Bind an LLM's explicit request to a current user quote, not its intent label alone.

    Everyday requests still use semantic planning. Pure questions about causes,
    deadlines or previous execution cannot authorize creating a draft.
    """
    if decision.action_request == "INFORMATION_ONLY":
        return False
    if authorizes_repair(message):
        return True
    quote = decision.repair_request_quote
    if not quote or quote not in message or decision.action_request != "EXPLICIT":
        return False
    question = re.search(r"[?？]|(?:吗|么|呢)[。！!\s]*$|什么时候|多久|为什么|为啥|是否|有没有|怎么|如何|怎样|能否", quote)
    request_form = re.search(r"(?:那就|就按).{0,16}(?:办|报|修|处理)|(?:报|开|建).{0,3}(?:一|个).{0,2}(?:单|修)|(?:让|叫|找).{0,10}(?:师傅|人).{0,8}(?:来|修|看|处理)", quote)
    return bool(request_form and not question)


def decision_context(session):
    from .operation_state import operation_snapshot
    from ..qa.day1_poc import configuration, session_root
    root = session_root(session)
    config = configuration(root) if root else None
    return {"knowledge_set": config["active_set"] if config else None,
            "question_set_id": config.get("question_set_id") if config else None, "history": session.dialogue_history[-8:], "repair_context": session.repair_dialogue,
            "has_pending_repair": bool(session.pending), "has_previous_order": bool(session.last_order),
            "free_condition_pending": session.repair_requires_free, "repair_details": session.repair_details,
            "last_reply_kind": session.last_reply_kind,
            "conversation_history": session.conversation_state["conversation_history"][-20:],
            "active_task": session.conversation_state["active_task"],
            "pending_conditions": session.conversation_state["pending_conditions"],
            "missing_slots": session.conversation_state["missing_slots"],
            "last_tool_result": session.conversation_state["last_tool_result"],
            "knowledge_context": session.conversation_state.get("knowledge_context"),
            "error_or_correction_state": session.conversation_state["error_or_correction_state"],
            "operation_state": operation_snapshot(session)}


def validate_decision(decision, message, context):
    from .llm import redact
    all_user_text = "\n".join([*context["history"], *context["repair_context"], message])
    for task in decision.tasks:
        if task.source_quote not in message and task.source_quote not in redact(message):
            raise POCError("INVALID_TOOL_INPUT", "暂时没能准确拆分您的问题，请重新说明；没有执行业务操作。", 400)
    for field, value in decision.repair.model_dump().items():
        # Slots were already verified when collected. Do not lose their provenance
        # merely because their original turn has left the bounded recent history.
        previously_verified = value == context["repair_details"].get(field)
        if value and not previously_verified and value not in all_user_text and value not in redact(all_user_text):
            raise POCError("INVALID_TOOL_INPUT", "未能核对故障信息，请重新说明；没有执行业务操作。", 400)
    for quote in (decision.condition_quote, decision.withdraw_condition_quote):
        if quote and quote not in all_user_text and quote not in redact(all_user_text):
            raise POCError("INVALID_TOOL_INPUT", "未能核对您提出的条件，请重新说明。", 400)
    if decision.repair_request_quote and decision.repair_request_quote not in message:
        raise POCError("INVALID_TOOL_INPUT", "未能核对本轮报修意愿，没有执行业务操作。", 400)
    if decision.route == "SERVICE" and not decision.tasks:
        raise POCError("INVALID_TOOL_PLAN", "请说明您想咨询或办理的事项。", 400)
    return decision


def execute_dialogue(session, decision, message, request_key):
    from .steward_actions import execute_steward_actions
    return execute_steward_actions(session, decision, message, request_key)




class LocalConversationPlanner:
    """Explicit offline fallback for demos/tests; not described as an online LLM."""
    label = "本地演示（未连接在线模型）"

    def decide(self, message, *, context):
        if CORRECTION.search(message):
            return DialogueDecision(route="CORRECTION")
        if CANCEL.search(message):
            return DialogueDecision(route="CANCEL")
        if any(t in message for t in ("餐厅", "好吃")) and not any(t in message for t in ("漏水", "报修", "维修")):
            return DialogueDecision(route="OUT_OF_SCOPE")
        if any(t in message.casefold() for t in ("chatgpt", "你好", "谢谢")):
            return DialogueDecision(route="SERVICE", tasks=[DialogueTask(intent="SOCIAL", source_quote=message)])
        if any(t in message for t in ("你不知道", "好笨", "你确定")):
            return DialogueDecision(route="SERVICE", tasks=[DialogueTask(intent="FEEDBACK", source_quote=message)])
        if context["repair_context"] and message.strip("。？? ") in {"地面", "墙面", "水槽下方"}:
            return DialogueDecision(route="SERVICE", tasks=[DialogueTask(intent="REPAIR_GUIDANCE", source_quote=message)],
                                    repair=RepairSlots(part=message.strip("。？? ")), use_repair_context=True)
        parts = [p.strip(" ，。？?！!") for p in re.split(r"另外|还有|顺便|以及", message) if p.strip(" ，。？?！!")]
        if len(parts) > 4:
            return DialogueDecision(route="CLARIFY")
        tasks = []
        for part in parts[:4]:
            old = LocalDemoPlanner().plan(part, last_order="available" if context["has_previous_order"] else None)
            fault = any(t in part for t in ("漏水", "滴水", "电梯又坏", "故障", "坏了", "不亮"))
            operation_question = re.search(r"(?:已经|有没有|是否|有).{0,10}(?:联系|通知|派单|提交|工单编号)|(?:进度|处理).{0,8}依据|(?:联系|通知).{0,10}(?:了|吗)", part)
            if operation_question and not authorizes_repair(part):
                subject = "EXTERNAL_CONTACT" if re.search(r"联系|通知", part) else "ORDER_RECEIPT"
                tasks.append(DialogueTask(intent="OPERATION_STATUS", source_quote=part, operation_subject=subject))
                continue
            if "投诉" in part and any(t in part for t in ("我要", "帮我", "申请", "如何", "怎么")):
                intent = "COMPLAINT"
            elif old["name"] == "track_order" or "工单进度" in part:
                intent = "TRACK_ORDER"
            elif authorizes_repair(part):
                intent = "REPAIR_PREPARE"
            elif context["repair_context"] and any(t in part for t in ("为什么", "为啥", "物业怎么搞")):
                intent = "REPAIR_CAUSE"
            elif fault and not any(t in part for t in ("报修渠道", "怎么报修", "如何报修", "维修流程", "收费", "费用", "保修期限")):
                intent = "REPAIR_GUIDANCE"
            elif FREE_CONDITION.search(part) and context["repair_context"]:
                intent = "REPAIR_PREPARE"
            else:
                intent = "KNOWLEDGE"
            tasks.append(DialogueTask(intent=intent, source_quote=part, query=part))
        slots = {}
        for key, regex in (("location", r"卫生间|厨房|水槽下方|墙面|地面|电梯|卧室"),
                           ("symptom", r"水管漏水|水龙头滴水|漏水|滴水|灯坏了|电梯又坏了|故障|坏了|不亮"),
                           ("started_at", r"昨晚开始|昨天开始|今天开始"),
                           ("available_time", r"(?:今晚|今天|明天).{0,8}(?:点|时).{0,3}(?:后|前)")):
            found = re.search(regex, message)
            if found:
                slots[key] = found.group(0)
        return DialogueDecision(route="SERVICE", tasks=tasks, repair=RepairSlots(**slots),
                                requires_free=bool(FREE_CONDITION.search(message)),
                                use_repair_context=bool(context["repair_context"] and not any(t in message for t in ("另一个故障", "另一个问题"))))
