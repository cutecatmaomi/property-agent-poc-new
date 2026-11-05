"""Private per-login graph state. A model cannot supply credentials, scope, confirmation or writes."""

import copy
import re
import secrets
from threading import RLock

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from ..domain import POCError
from ..response_policy import classify_message, normalized_text
from .conversation import build_conversation_graph, new_state, remember
from .planner import LocalDemoPlanner
from .presentation import error_reply
from .tools import ControlledTools
from .workflow import build_graph

ESTATES = {"E001": "烟台港城华府", "E002": "青岛国际健康生态谷"}


class AgentSession:
    def __init__(self, actions, token, *, planner=None, owns_actions=False):
        self.actions, self.token = actions, token
        self.planner = planner or LocalDemoPlanner()
        self._owns_actions = owns_actions
        who = actions.auth.resolve(token)
        self._binding = (who.estate_id, who.actor_ref, who.session_hash)
        self.estate_id = who.estate_id
        self.session_id = secrets.token_hex(16)
        self.thread_id = self.session_id
        self.conversation_state = new_state(self.session_id, who.actor_ref, who.estate_id)
        self._planned_decision = None
        self._inline_decision = None
        self._closed = False
        self.context = actions.resident_context(token)
        self.unit_id = next((u["unit_id"] for u in self.context["units"] if u["can_create_work_order"]),
                            self.context["units"][0]["unit_id"] if self.context["units"] else None)
        self.knowledge_mode, self.renovation_activity = "demo", None
        self.last_order = None
        self.dialogue_history = []
        self.repair_dialogue = []
        self.repair_details = {}
        self.repair_requires_free = False
        self.last_reply_kind = None
        self.pending_requires_recheck = False
        self._lock = RLock()
        self._pending = None
        self._turns = {}
        self.tools = ControlledTools(actions, token)
        self._memory = InMemorySaver()
        self._graph = build_graph(self, self._memory)
        self._conversation_memory = InMemorySaver()
        self._conversation_graph = build_conversation_graph(self, self._conversation_memory)
        from ..qa.day1_poc import knowledge_epoch
        self.knowledge_epoch = knowledge_epoch(self)

    def verify(self):
        if self._closed:
            raise POCError("UNAUTHENTICATED", "请重新登录", 401)
        who = self.actions.auth.resolve(self.token)
        if (who.estate_id, who.actor_ref, who.session_hash) != self._binding:
            raise POCError("UNAUTHENTICATED", "登录身份已变化，请重新登录", 401)
        self.context = self.actions.resident_context(self.token)
        if self.unit_id and self.unit_id not in {u["unit_id"] for u in self.context["units"]}:
            raise POCError("RESOURCE_NOT_AVAILABLE", "房屋关联已失效，请重新登录核验", 404)
        from ..qa.day1_poc import knowledge_epoch
        epoch = knowledge_epoch(self)
        if epoch != getattr(self, 'knowledge_epoch', None):
            # Retire historical knowledge, not drafts, confirmation nonces or receipts.
            self.conversation_state['conversation_history'] = []
            self.conversation_state.update(knowledge_context=None, knowledge_evidence=[], final_answer='')
            self.dialogue_history = []
            self._conversation_memory.delete_thread(self.thread_id)
            self.knowledge_epoch = epoch

    def foreign_scope(self, message):
        mentions = set(re.findall(r"(?<![A-Z0-9])E00[12](?![A-Z0-9])", message.upper()))
        mentions.update(e for e, name in ESTATES.items() if name in message)
        return bool(mentions - {self.estate_id})

    def set_context(self, unit_id, knowledge_mode, renovation_activity=None):
        with self._lock:
            self.verify()
            if knowledge_mode not in {"demo", "current"} or renovation_activity not in {
                None, "BARE", "MINOR", "DEMOLITION", "FLOOR_HEATING",
            }:
                raise POCError("INVALID_INPUT", "用途或装修活动无效", 400)
            if unit_id and unit_id not in {u["unit_id"] for u in self.context["units"]}:
                raise POCError("RESOURCE_NOT_AVAILABLE", "房屋不存在或无权操作", 404)
            if (unit_id, knowledge_mode, renovation_activity) == (self.unit_id, self.knowledge_mode, self.renovation_activity):
                return False
            self._cancel_pending()
            for key in self._turns:
                self._memory.delete_thread(key)
            self._turns.clear()
            self.last_order = None
            self.dialogue_history.clear()
            self.repair_dialogue.clear()
            self.repair_details.clear()
            self.repair_requires_free = False
            self.last_reply_kind = None
            self.pending_requires_recheck = False
            self.unit_id, self.knowledge_mode, self.renovation_activity = unit_id, knowledge_mode, renovation_activity
            self.conversation_state = new_state(self.session_id, self._binding[1], self.estate_id)
            self._conversation_memory.delete_thread(self.thread_id)
            return True

    @property
    def pending(self):
        preview = copy.deepcopy(self._turns[self._pending].get("preview_reply")) if self._pending else None
        if preview and self.pending_requires_recheck:
            preview["blocked_reason"] = "刚刚补充的信息未能核对，旧卡片暂不能提交；请重新说明报修内容或取消。"
        return preview

    def valid_resume(self, request_key, answer):
        turn = self._turns.get(request_key)
        return bool(turn and isinstance(answer, dict) and type(answer.get("accept")) is bool
                    and answer.get("nonce") == turn["nonce"] and answer["accept"] is turn["accepted"]
                    and self._pending == request_key)

    @staticmethod
    def _config(key):
        return {"configurable": {"thread_id": key}, "recursion_limit": 16}

    def _compact_conversation(self):
        # Finished conversation turns have no interrupt. Keep one current checkpoint,
        # rather than accumulating all historical copies of the bounded message list.
        self._conversation_memory.delete_thread(self.thread_id)
        self._conversation_graph.update_state({"configurable": {"thread_id": self.thread_id}},
            copy.deepcopy(self.conversation_state), as_node="remember")

    def _finish(self, key, state):
        turn = self._turns[key]
        if state.get("__interrupt__"):
            self._pending = key
            self.pending_requires_recheck = False
            value = state["__interrupt__"][0].value
            reply = {"kind": "REPAIR_PREVIEW", "text": value.get("assistant_text") or "请核对下方房屋和故障描述，10分钟内点击按钮确认；当前尚未创建工单。",
                     "ticket": key, "data": value, "trace": state["trace"]}
            turn["preview_reply"] = reply
            return copy.deepcopy(reply)
        reply = {**state["reply"], "trace": state["trace"]}
        if self._pending == key:
            self._pending = None
            self.pending_requires_recheck = False
        if reply["kind"] == "ORDER_CREATED":
            self.last_order = reply["data"]["number"]
            turn["reply"] = reply
            # Keep a bounded public receipt for repeat clicks; no model or graph state is exposed.
            completed = [k for k, t in self._turns.items() if t.get("reply")]
            for old in completed[:-20]:
                del self._turns[old]
        else:
            del self._turns[key]
        self._memory.delete_thread(key)
        return copy.deepcopy(reply)

    def _cancel_pending(self):
        if not self._pending:
            return
        if self._turns[self._pending]["accepted"] is True:
            raise POCError("SUBMISSION_PENDING", "提交结果待确认，请先点击同一张报修卡片重试，勿重复准备报修", 409)
        reply = self.confirm(self._pending, accept=False, _internal=True)
        if reply["kind"] == "ERROR" and reply["data"]["status"] != 410:
            raise POCError(reply["data"]["error"], reply["text"], reply["data"]["status"])

    def chat(self, message):
        with self._lock:
            try:
                self.verify()
                if not isinstance(message, str) or not 1 <= len(message.strip()) <= 300:
                    raise POCError("INVALID_INPUT", "请填写1至300字的问题或故障描述", 400)
                message = message.strip()
                if hasattr(self.planner, "decide"):
                    from .dialogue import CANCEL, CORRECTION, FREE_CONDITION, execute_dialogue
                    from .dialogue_contract import DialogueDecision
                    # Stop stale actions even if the model is down or produces an invalid response.
                    if not classify_message(message) and not self.foreign_scope(message):
                        if CORRECTION.search(message) or CANCEL.search(message):
                            if not getattr(self.planner, "enable_natural_reply", False):
                                result = execute_dialogue(self, DialogueDecision(route="CORRECTION" if CORRECTION.search(message) else "CANCEL"), message, "")
                                from .presentation import present
                                reply = present(result)
                                remember(self, message, reply)
                                return reply
                            # Feedback about the last knowledge answer must not cancel
                            # an unrelated repair draft. Semantic execution still
                            # invalidates drafts when actual repair slots change.
                            knowledge_feedback = (CORRECTION.search(message) and not CANCEL.search(message)
                                and self.conversation_state.get("current_topic") == "KNOWLEDGE"
                                and bool(self.conversation_state.get("knowledge_context")))
                            if not knowledge_feedback:
                                self._cancel_pending()
                        if FREE_CONDITION.search(message):
                            self.repair_requires_free = True
                            self._cancel_pending()
                # A submission with an unknown result can only retry its original trusted confirmation.
                if self._pending and self._turns[self._pending]["accepted"] is True and not classify_message(message):
                    raise POCError("SUBMISSION_PENDING", "提交结果待确认，请先点击同一张报修卡片重试，勿重复准备报修", 409)
                if self._pending and normalized_text(message) in {"确认", "确认提交", "提交", "yes", "confirm"}:
                    return {"kind": "CLARIFY", "text": "请点击当前报修卡片的“确认提交报修”按钮；聊天文字不会提交工单。", "data": {}}
                # Safety replies remain available even when a previous submission needs a retry.
                from ..qa.day1_poc import exact_question
                if not hasattr(self.planner, "decide") and not exact_question(self, message) and not (classify_message(message) and self._pending and self._turns[self._pending]["accepted"] is True):
                    self._cancel_pending()
                key = secrets.token_hex(16)
                self._turns[key] = {"nonce": secrets.token_urlsafe(24), "accepted": None}
                try:
                    self.conversation_state["request_key"] = key
                    state = self._conversation_graph.invoke({**copy.deepcopy(self.conversation_state), "current_user_message": message,
                        "request_key": key}, {"configurable": {"thread_id": self.thread_id}, "recursion_limit": 16})
                except BaseException:
                    self._memory.delete_thread(key)
                    self._turns.pop(key, None)
                    raise
                self._compact_conversation()
                return state["reply"]
            except POCError as exc:
                if hasattr(self.planner, "decide") and self._pending and self._turns[self._pending]["accepted"] is None:
                    self.pending_requires_recheck = True
                return error_reply(exc)

    def confirm(self, ticket, *, accept, _internal=False):
        """Trusted UI operation, deliberately absent from all model-facing schemas/tools."""
        with self._lock:
            try:
                self.verify()
                if type(accept) is not bool or ticket not in self._turns:
                    raise POCError("CONFIRMATION_REQUIRED", "确认卡片已失效，请重新准备报修", 409)
                turn = self._turns[ticket]
                if turn.get("reply"):
                    if not accept:
                        raise POCError("ALREADY_SUBMITTED", "报修已经提交，请通过工单进度查看处理情况", 409)
                    reply = copy.deepcopy(turn["reply"])
                    reply["data"]["replayed"] = True
                    reply["text"] += "\n重复确认：返回原工单，未新增工单。"
                    return reply
                if ticket != self._pending or turn["accepted"] is not None and turn["accepted"] is not accept:
                    raise POCError("CONFIRMATION_REQUIRED", "请使用当前报修卡片重试相同操作", 409)
                if accept and self.repair_requires_free:
                    raise POCError("CONDITION_UNRESOLVED", "维修费用尚未核实，不能按“免费才报修”的条件提交。", 409)
                if accept and any(c["status"] == "UNRESOLVED" for c in self.conversation_state["pending_conditions"]):
                    raise POCError("CONDITION_UNRESOLVED", "你提出的报修前提尚未满足，请先核对条件。", 409)
                if accept and self.pending_requires_recheck:
                    raise POCError("CONDITION_UNRESOLVED", "刚刚补充的信息未能核对，请重新说明报修内容后再确认。", 409)
                turn["accepted"] = accept
                config = self._config(ticket)
                snapshot = self._graph.get_state(config)
                command = Command(resume={"accept": accept, "nonce": turn["nonce"]}) if snapshot.next == ("approval",) else None
                reply = self._finish(ticket, self._graph.invoke(command, config))
                if not _internal:
                    remember(self, "点击确认提交报修" if accept else "点击取消本次报修", reply, event=True)
                    self._compact_conversation()
                return reply
            except POCError as exc:
                if exc.status == 410 and ticket == self._pending:
                    self._pending = None
                    self._turns.pop(ticket, None)
                    self._memory.delete_thread(ticket)
                return error_reply(exc)

    def close(self):
        with self._lock:
            if self._closed:
                return
            try:
                if self._pending and self._turns[self._pending]["accepted"] is None:
                    self.confirm(self._pending, accept=False)
                self.actions.logout(self.token)
            except POCError:
                pass
            finally:
                self._closed = True
                for key in self._turns:
                    self._memory.delete_thread(key)
                self._turns.clear()
                self._pending, self.last_order = None, None
                self.dialogue_history.clear()
                self.repair_dialogue.clear()
                self.repair_details.clear()
                self.repair_requires_free = False
                self.last_reply_kind = None
                self.pending_requires_recheck = False
                self.conversation_state = new_state(self.session_id, self._binding[1], self.estate_id)
                self._conversation_memory.delete_thread(self.thread_id)
                if callable(close_planner := getattr(self.planner, "close", None)):
                    close_planner()
                if self._owns_actions:
                    self.actions.close()
