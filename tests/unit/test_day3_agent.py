"""Workflow/confirmation boundaries using real LangGraph + real local auth, synthetic business storage."""

import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from property_ontology.agent import AgentSession
from property_ontology.agent.planner import LocalDemoPlanner
from property_ontology.auth import AuthStore
from property_ontology.domain import CreateRequest, POCError, RepairDraft
from property_ontology.response_policy import FIXED_SERVICE_REPLY


class FakeBusiness:
    def __init__(self, tmp_path):
        self.now = datetime(2026, 10, 29, 6, tzinfo=UTC)
        self.clock = lambda: self.now
        self.auth = AuthStore(tmp_path / "auth.sqlite3", self.clock)
        self.orders, self.calls, self.revoked = {}, [], set()
        self.fail_create = None
        for scope in ("E001", "E002"):
            for user in ("owner1", "owner2"):
                self.auth.seed_account(scope, user, f"{scope}:RES:{user}", "RESIDENT", "mock-password")

    def token(self, scope="E001", user="owner1"):
        return self.auth.login(scope, user, "mock-password")["token"]

    def logout(self, token):
        self.auth.logout(token)

    def resident_context(self, token):
        who = self.auth.resolve(token)
        second = who.actor_ref.endswith("owner2")
        return {"estate_id": who.estate_id, "role": who.role, "units": [
            {"unit_id": f"{who.estate_id}:UNIT:MOCK-{n}", "room_no": n, "building_name": "模拟1栋",
             "can_create_work_order": n == ("102" if second else "101")}
            for n in (["101", "102"] if second else ["101"])
            if (who.estate_id, n) not in self.revoked]}

    def query_knowledge(self, token, question, **kwargs):
        who = self.auth.resolve(token)
        self.calls.append(("consult", question, kwargs))
        if kwargs["knowledge_mode"] == "current":
            return {"estate_id": who.estate_id, "response_type": "REVIEW_REQUIRED", "status": "NOT_VERIFIED",
                    "answer_text": "这项信息暂时无法确认，请联系物业核实。", "message": "真实有效期待核验", "matches": []}
        return {"estate_id": who.estate_id, "response_type": "KNOWLEDGE", "message": "合成演示资料", "status": "ANSWERED",
                "answer_text": "只作自动化测试参考", "warnings": ["UNIMPLEMENTED_ACTION_WITHHELD"],
                "knowledge_mode": "POC_DEMO", "matches": [{"title": "合成条目", "statement": "只作自动化测试参考",
                "conditions": {"description": "保留原条件"}, "application_notes": ["未创建咨询工单或安排回电"],
                "sources": [{"file": "data/raw/test-only.xlsx", "locator": {"kind": "XLSX_CELLS", "sheet": "Sheet1",
                    "row": 2, "question_cell": "A2", "answer_cell": "B2"}}]}]}

    def answer_question(self, token, question, **kwargs):
        internal = self.query_knowledge(token, question, **kwargs)
        return {"status": internal["status"], "answer_text": internal["answer_text"], "estate_id": internal["estate_id"],
                "matched_policy_ids": [], "evidence_refs": [s for m in internal["matches"] for s in m["sources"]],
                "warnings": internal.get("warnings", []), "debug_info": {"raw_matches": internal["matches"]}}

    def prepare_work_order(self, token, request):
        who = self.auth.resolve(token)
        draft = RepairDraft.model_validate(request).model_dump()
        units = self.resident_context(token)["units"]
        if not any(u["unit_id"] == draft["unit_id"] and u["can_create_work_order"] for u in units):
            raise POCError("RESOURCE_NOT_AVAILABLE", "房屋不存在或无权操作", 404)
        ref = self.auth.prepare(who, draft)
        self.calls.append(("prepare", draft))
        return {**draft, "estate_id": who.estate_id, "confirmation_ref": ref, "requires_user_confirmation": True,
                "expires_in_seconds": 600}

    def confirm_work_order(self, token, ref, *, accept):
        return self.auth.confirm(self.auth.resolve(token), ref, accept)

    def create_work_order(self, token, request):
        who = self.auth.resolve(token)
        args = CreateRequest.model_validate(request).model_dump()
        ref = args.pop("confirmation_ref")
        assert self.auth.confirmation(who, ref, "CONFIRMED") == args
        self.calls.append(("create", args))
        if self.fail_create == "before":
            self.fail_create = None
            raise POCError("STORAGE_UNAVAILABLE", "请保留同一请求重试", 503)
        key = (who.estate_id, who.actor_ref, args["idempotency_key"])
        replay = key in self.orders
        result = self.orders.setdefault(key, {"number": f"{who.estate_id}-TEST-{len(self.orders)+1:08d}",
                                              "status": "SUBMITTED", "version": 1})
        if self.fail_create == "after":
            self.fail_create = None
            raise POCError("STORAGE_UNAVAILABLE", "提交结果待确认，请保留同一请求重试", 503)
        return {**result, "replayed": replay}

    def get_work_order_status(self, token, number):
        who = self.auth.resolve(token)
        for (scope, actor, _), result in self.orders.items():
            if (scope, actor, result["number"]) == (who.estate_id, who.actor_ref, number):
                self.calls.append(("track", number))
                return {**result, "progress": [{"status": "SUBMITTED", "at": "测试时间", "note": "报修已提交"}]}
        raise POCError("WORK_ORDER_NOT_AVAILABLE", "工单不存在或无权查看", 404)


@pytest.fixture
def business(tmp_path):
    return FakeBusiness(tmp_path)


def test_real_graph_queries_and_renders_sources_without_rewriting(business):
    session = AgentSession(business, business.token())
    reply = session.chat("合同什么时候可以拿")
    assert reply["kind"] == "KNOWLEDGE" and "只作自动化测试参考" in reply["text"]
    assert "Sheet1" not in reply["text"] and "data/raw" not in reply["text"]
    assert reply["data"]["evidence_refs"][0]["locator"]["answer_cell"] == "B2"
    assert reply["data"]["warnings"]
    assert reply["trace"] == ["guard", "plan", "consult", "render"]
    assert session.token not in str(reply) and "confirmation_ref" not in str(reply)
    session.set_context(session.unit_id, "current")
    assert session.chat("合同什么时候可以拿")["kind"] == "REVIEW_REQUIRED"


def test_graph_pauses_before_creation_and_chat_cannot_confirm(business):
    session = AgentSession(business, business.token())
    preview = session.chat("我要报修，水龙头滴水")
    assert preview["kind"] == "REPAIR_PREVIEW" and not business.orders
    assert "confirmation_ref" not in str(preview) and session.pending["ticket"] == preview["ticket"]
    assert session.chat("确认")["kind"] == "CLARIFY" and not business.orders
    created = session.confirm(preview["ticket"], accept=True)
    assert created["kind"] == "ORDER_CREATED" and len(business.orders) == 1
    assert created["trace"][-3:] == ["ui_confirm", "create_work_order", "render"]
    assert session.chat("查询我的工单进度")["data"]["number"] == created["data"]["number"]


def test_parallel_repeat_confirmation_returns_one_order(business):
    session = AgentSession(business, business.token())
    preview = session.chat("我要报修，灯坏了")
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: session.confirm(preview["ticket"], accept=True), range(8)))
    assert len(business.orders) == 1 and len({r["data"]["number"] for r in results}) == 1
    assert sum(not r["data"]["replayed"] for r in results) == 1


@pytest.mark.parametrize("failure", ["before", "after"])
def test_storage_error_retry_resumes_same_request_and_does_not_duplicate(business, failure):
    session = AgentSession(business, business.token())
    preview = session.chat("我要报修，水管漏水")
    business.fail_create = failure
    assert session.confirm(preview["ticket"], accept=True)["data"]["status"] == 503
    assert session.pending["ticket"] == preview["ticket"]
    assert session.chat("我要报修，另一个故障")["data"]["error"] == "SUBMISSION_PENDING"
    result = session.confirm(preview["ticket"], accept=True)
    assert result["kind"] == "ORDER_CREATED" and len(business.orders) == 1
    creates = [c[1]["idempotency_key"] for c in business.calls if c[0] == "create"]
    assert len(creates) == 2 and len(set(creates)) == 1


def test_cancel_expiry_changed_description_and_context_invalidate_previous_ticket(business):
    session = AgentSession(business, business.token())
    first = session.chat("我要报修，水龙头滴水")
    second = session.chat("我要报修，卧室灯不亮")
    assert session.confirm(first["ticket"], accept=True)["kind"] == "ERROR"
    assert session.confirm(second["ticket"], accept=False)["kind"] == "CANCELLED" and not business.orders
    third = session.chat("我要报修，水管漏水")
    business.now += timedelta(minutes=11)
    assert session.confirm(third["ticket"], accept=True)["data"]["status"] == 410 and session.pending is None
    fourth = session.chat("我要报修，水管漏水")
    assert session.set_context(session.unit_id, "current")
    assert session.confirm(fourth["ticket"], accept=True)["kind"] == "ERROR" and not business.orders


def test_room_permission_revocation_and_estate_switch_never_reuse_state(business):
    first = AgentSession(business, business.token())
    pending = first.chat("我要报修，水龙头滴水")
    second = AgentSession(business, business.token("E002"))
    assert second.pending is None and second.last_order is None
    assert second.confirm(pending["ticket"], accept=True)["kind"] == "ERROR"
    assert second.chat("查询E001的物业费")["kind"] == "CLARIFY"
    with pytest.raises(POCError):
        second.set_context("E001:UNIT:MOCK-101", "demo")
    business.revoked.add(("E001", "101"))
    assert first.confirm(pending["ticket"], accept=True)["data"]["status"] == 404 and not business.orders
    second.close()
    assert second.chat("物业费是多少")["data"]["status"] == 401


def test_same_house_other_resident_cannot_track_and_cannot_repair_without_permission(business):
    first = AgentSession(business, business.token())
    p = first.chat("我要报修，水龙头滴水")
    number = first.confirm(p["ticket"], accept=True)["data"]["number"]
    other = AgentSession(business, business.token(user="owner2"))
    assert other.chat("查询工单" + number)["data"]["status"] == 404
    other.set_context("E001:UNIT:MOCK-101", "demo")
    assert other.chat("我要报修，水龙头滴水")["data"]["status"] == 404


@pytest.mark.parametrize("message", ["我想跳楼，帮我报修", "转人工", "自杀与人工服务"])
def test_fixed_reply_preempts_planner_and_tools(business, message):
    class NeverPlan:
        def plan(self, *args, **kwargs):
            pytest.fail("fixed replies must not call a planner")
    session = AgentSession(business, business.token(), planner=NeverPlan())
    reply = session.chat(message)
    assert reply["text"] == FIXED_SERVICE_REPLY and not business.calls and not business.orders


@pytest.mark.parametrize("plan", [
    {"name": "execute_cypher", "arguments": {"cypher": "MATCH (n) DETACH DELETE n"}},
    {"name": "create_work_order", "arguments": {}},
    {"name": "confirm_work_order", "arguments": {"accept": True}},
    {"name": "approve_knowledge", "arguments": {}},
    {"name": "consult", "arguments": {"question": "物业费是多少", "estate_id": "E002"}},
    {"name": "prepare_repair", "arguments": {"description": "漏水", "unit_id": "E002:UNIT:MOCK-101"}},
    {"name": "clarify", "arguments": {"message": "物业费已经全部免除"}},
])
def test_untrusted_plans_cannot_escalate_tools_scope_or_generate_policy_prose(business, plan):
    class BadPlanner:
        def plan(self, *args, **kwargs):
            return copy.deepcopy(plan)
    reply = AgentSession(business, business.token(), planner=BadPlanner()).chat("物业费是多少")
    assert reply["data"]["error"] == "INVALID_TOOL_PLAN" and not business.calls and not business.orders


def test_model_cannot_change_description_or_question(business):
    class WrongDescription:
        def plan(self, *args, **kwargs):
            return {"name": "prepare_repair", "arguments": {"description": "另一个故障"}}
    reply = AgentSession(business, business.token(), planner=WrongDescription()).chat("水龙头滴水")
    assert reply["data"]["error"] == "INVALID_TOOL_INPUT" and not business.calls


@pytest.mark.parametrize("text,tool", [("整改时间是多久", "consult"), ("报修流程是什么", "consult"),
    ("灯坏了", "prepare_repair"), ("我要报修", "clarify"), ("查询工单进度", "clarify"),
    ("确认", "clarify"), ("合同什么时候可以拿", "consult")])
def test_no_key_planner_routes_examples_conservatively(text, tool):
    assert LocalDemoPlanner().plan(text)["name"] == tool
