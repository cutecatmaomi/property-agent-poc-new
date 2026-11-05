"""Controlled business actions shared by the Day2 CLI and Day3 workflow; no arbitrary Cypher interface."""

import json
import secrets
import sqlite3

from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired
from pydantic import ValidationError

from .auth import AuthStore
from .domain import (
    CreateRequest,
    POCError,
    RepairDraft,
    UpdateRequest,
    canonical,
    digest,
    utcnow,
)
from .qa.contract import AnswerResult
from .qa.service import KnowledgeAnswers
from .repository import GraphRepository
from .response_policy import (
    classify_message,
    fixed_reply,
)


class PropertyActions:
    def __init__(self, settings, *, clock=utcnow):
        self.settings, self.clock = settings, clock
        self.auth = AuthStore(settings.auth_path, clock)
        self._repo = GraphRepository(settings)

    def close(self):
        self._repo.close()

    def login(self, estate_id, username, password):
        try:
            result = self.auth.login(estate_id, username, password)
            who = self.auth.resolve(result["token"])
            if who.role == "RESIDENT" and not self._repo.resident_active(who.estate_id, who.actor_ref):
                self.auth.logout(result["token"])
                raise POCError("UNAUTHENTICATED", "登录验证失败", 401)
            return result
        except (Neo4jError, ServiceUnavailable, SessionExpired, sqlite3.Error) as exc:
            raise POCError("STORAGE_UNAVAILABLE", "登录服务暂不可用", 503) from exc

    def logout(self, session_token):
        self.auth.logout(session_token)
        return {"logged_out": True}

    def resident_context(self, session_token):
        return self._run(session_token, "resident_context", "RESIDENT", lambda who: {
            "estate_id": who.estate_id, "role": who.role,
            "units": self._repo.resident_units(who, self.clock().date()),
        })

    def _run(self, token, action, role, function):
        request_id = secrets.token_hex(12)
        try:
            who = self.auth.resolve(token)
        except POCError:
            scope = token.split(".", 1)[0] if isinstance(token, str) else None
            if scope in {"E001", "E002"}:
                self.auth.audit(scope, None, action, "UNAUTHENTICATED", request_id)
            raise
        except sqlite3.Error as exc:
            raise POCError("STORAGE_UNAVAILABLE", "身份服务暂不可用", 503) from exc
        try:
            if role and who.role != role:
                raise POCError("ROLE_REQUIRED", "当前身份无权执行此操作", 403)
            if who.role == "RESIDENT" and not self._repo.resident_active(who.estate_id, who.actor_ref):
                raise POCError("UNAUTHENTICATED", "请重新登录", 401)
            result = function(who)
            self.auth.audit(who.estate_id, who.actor_ref, action, "OK", request_id)
            return result
        except ValidationError as exc:
            self.auth.audit(who.estate_id, who.actor_ref, action, "INVALID_INPUT", request_id)
            raise POCError("INVALID_INPUT", "输入参数不符合接口要求", 400) from exc
        except POCError as exc:
            diagnostics = getattr(exc, "diagnostics", None)
            self.auth.audit(who.estate_id, who.actor_ref, action,
                            canonical({"error": exc.code, "diagnostics": diagnostics}) if diagnostics else exc.code, request_id)
            raise
        except (Neo4jError, ServiceUnavailable, SessionExpired, sqlite3.Error) as exc:
            raise POCError("STORAGE_UNAVAILABLE", "服务暂不可用，请保留幂等键后重试", 503) from exc

    @staticmethod
    def _payload(who, draft, service_id):
        return {
            "estate_id": who.estate_id,
            "resident_id": who.actor_ref,
            "unit_id": draft.unit_id,
            "service_id": service_id,
            "description": draft.description,
            "idempotency_key": draft.idempotency_key,
        }

    def prepare_work_order(self, session_token, request):
        def prepare(who):
            draft = RepairDraft.model_validate(request)
            reason = classify_message(draft.description)
            if reason:
                return fixed_reply(who.estate_id, reason)
            eligible = self._repo.eligible_unit(who, draft.unit_id, self.clock().date())
            payload = self._payload(who, draft, eligible["service_id"])
            ref = self.auth.prepare(who, payload)
            return {
                "confirmation_ref": ref,
                "estate_id": who.estate_id,
                "unit_id": draft.unit_id,
                "description": draft.description,
                "service_code": "maintenance",
                "idempotency_key": draft.idempotency_key,
                "payload_hash": digest(payload),
                "requires_user_confirmation": True,
                "expires_in_seconds": 600,
            }

        return self._run(session_token, "prepare_work_order", "RESIDENT", prepare)

    def confirm_work_order(self, session_token, confirmation_ref, *, accept):
        # This is a trusted user-interface operation, NOT a future LLM tool.
        def confirm(who):
            payload = self.auth.confirmation(who, confirmation_ref)
            if payload.get("request_type", "REPAIR") != "REPAIR":
                raise POCError("CONFIRMATION_MISMATCH", "旧人工请求已停用，请重新准备报修", 409)
            reason = classify_message(payload["description"])
            if reason:
                return fixed_reply(who.estate_id, reason)
            self._repo.eligible_unit(who, payload["unit_id"], self.clock().date())
            return self.auth.confirm(who, confirmation_ref, accept)

        return self._run(session_token, "confirm_work_order", "RESIDENT", confirm)

    def _mirror(self, order):
        try:
            self.auth.mirror_events(
                order["estate_id"], order["work_order_id"], json.loads(order["state_history_json"])
            )
            return False
        except sqlite3.Error:
            # Durable history/outbox is already part of the Neo4j transaction. Read/replay resynchronizes it.
            return True

    def create_work_order(self, session_token, request):
        def create(who):
            args = CreateRequest.model_validate(request)
            reason = classify_message(args.description)
            if reason:
                return fixed_reply(who.estate_id, reason)
            confirmed = self.auth.confirmation(who, args.confirmation_ref, "CONFIRMED")
            eligible = self._repo.eligible_unit(who, args.unit_id, self.clock().date())
            payload = self._payload(who, args, eligible["service_id"])
            if confirmed != payload:
                raise POCError("CONFIRMATION_MISMATCH", "报修内容已改变，请重新确认", 409)
            fingerprint = digest({k: v for k, v in payload.items() if k != "idempotency_key"})
            order, replayed = self._repo.create_order(
                who, payload, fingerprint, digest(payload), self.clock()
            )
            pending = self._mirror(order)
            return {
                **{
                    k: order[k]
                    for k in ["work_order_id", "number", "status", "created_at", "updated_at", "version"]
                },
                "replayed": replayed,
                "event_mirror_pending": pending,
            }

        return self._run(session_token, "create_work_order", "RESIDENT", create)

    def get_work_order_status(self, session_token, number):
        def get(who):
            if not isinstance(number, str) or len(number) > 160:
                raise POCError("INVALID_INPUT", "工单编号无效", 400)
            order = self._repo.order_for_resident(who, number, self.clock().date())
            self._mirror(order)
            history = json.loads(order["state_history_json"])
            result = {
                **{k: order[k] for k in ["number", "status", "created_at", "updated_at", "version"]},
                "progress": [
                    {"status": e["new_status"], "at": e["at"], "note": e.get("public_note", e["new_status"])}
                    for e in history
                ],
            }
            return result

        return self._run(session_token, "get_work_order_status", "RESIDENT", get)

    def update_work_order_status(self, session_token, request):
        def update(who):
            args = UpdateRequest.model_validate(request)
            order, event, replayed = self._repo.update_order(
                who, args, digest(args.model_dump()), self.clock()
            )
            pending = self._mirror(order)
            return {
                "number": order["number"],
                "previous_status": event["old_status"],
                "status": event["new_status"],
                "updated_at": event["at"],
                "version": event["result_version"],
                "event_id": event["event_id"],
                "replayed": replayed,
                "event_mirror_pending": pending,
            }

        return self._run(session_token, "update_work_order_status", "BACKOFFICE_OPERATOR", update)

    def answer_question(self, session_token, question, **kwargs):
        """Versioned AnswerResult contract, shared by CLI and Agent."""
        result = self.query_knowledge(session_token, question, **kwargs)
        return AnswerResult.model_validate({k: result[k] for k in AnswerResult.model_fields}).model_dump()

    def query_knowledge(
        self,
        session_token,
        question,
        *,
        service_code=None,
        unit_id=None,
        knowledge_mode="current",
        renovation_activity=None,
        semantic_matcher=None,
        semantic_context=None,
    ):
        def query(who):
            result = KnowledgeAnswers(self._repo, self.auth, self.settings.root, self.clock().date()).answer(
                who, question, service_code=service_code, unit_id=unit_id,
                knowledge_mode=knowledge_mode, renovation_activity=renovation_activity,
                semantic_matcher=semantic_matcher, semantic_context=semantic_context,
            )
            self.auth.record_answer(who.estate_id, who.actor_ref, result)
            return result

        return self._run(session_token, "query_knowledge", None, query)
