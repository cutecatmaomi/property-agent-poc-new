"""HTML v3 failures reproduced through the real workflow, with explicitly scripted model responses."""

import copy
import json

import httpx
import pytest
from test_day3_agent import FakeBusiness

from property_ontology.agent import AgentSession
from property_ontology.agent.dialogue import LocalConversationPlanner
from property_ontology.agent.dialogue_contract import strict_schema
from property_ontology.agent.llm import LLMPlanner, LLMSettings, build_planner
from property_ontology.domain import POCError
from property_ontology.response_policy import FIXED_SERVICE_REPLY


@pytest.fixture
def agent(tmp_path):
    business = FakeBusiness(tmp_path)
    session = AgentSession(business, business.token(), planner=LocalConversationPlanner())
    yield session, business
    session.close()


def test_v3_leak_help_and_angry_followup_keep_repair_context_without_complaint(agent):
    session, business = agent
    message = "昨晚开始厨房就一直滴滴答答漏水，我也不知道哪里坏了，怎么办？"
    first = session.chat(message)
    assert "不能确定漏水原因" in first["text"] and first["text"].count("？") == 1
    assert session.pending is None and not business.orders and not business.calls
    second = session.chat("为啥漏水啊？你们物业怎么搞的")
    assert "现场检查" in second["text"] and "投诉" not in second["text"]
    assert session.repair_dialogue[0] == message and not business.orders


def test_v3_correction_stops_old_draft_even_if_model_is_unavailable(agent):
    session, business = agent
    p = session.chat("卫生间水管漏水，今晚七点后有人在家，直接帮我报修。")
    assert p["kind"] == "REPAIR_PREVIEW" and p["data"]["dialogue_details"]["available_time"] == "今晚七点后"

    class Down:
        def decide(self, *args, **kwargs):
            pytest.fail("explicit correction must not depend on a model request")

    session.planner = Down()
    answer = session.chat("你创建错了吧")
    assert answer["kind"] == "CORRECTED" and session.pending is None
    assert session.confirm(p["ticket"], accept=True)["kind"] == "ERROR" and not business.orders


def test_v3_free_condition_revokes_prior_card_and_cannot_be_confirmed(agent):
    session, business = agent
    p = session.chat("卫生间水管漏水，今晚七点后有人在家，直接帮我报修。")
    result = session.chat("这个维修收费吗？如果免费就帮我报修")
    assert "无法确认" in result["text"] and session.repair_requires_free
    assert session.pending is None and not business.orders
    assert session.confirm(p["ticket"], accept=True)["kind"] == "ERROR"
    session.chat("帮我报修")
    assert session.pending is None and not business.orders
    p2 = session.chat("收费也可以，帮我报修")
    assert p2["kind"] == "REPAIR_PREVIEW" and not session.repair_requires_free
    assert "今晚七点后" in p2["data"]["description"]
    created = session.confirm(p2["ticket"], accept=True)
    assert created["kind"] == "ORDER_CREATED" and len(business.orders) == 1
    assert session.confirm(p2["ticket"], accept=True)["data"]["replayed"]


def test_v3_free_condition_blocks_even_when_model_fails(agent):
    session, business = agent
    p = session.chat("帮我报修，水管漏水")

    class Down:
        def decide(self, *args, **kwargs):
            raise POCError("LLM_UNAVAILABLE", "模型连接失败", 503)

    session.planner = Down()
    assert session.chat("如果免费才报修")["kind"] == "ERROR"
    assert session.pending is None and session.repair_requires_free
    assert session.confirm(p["ticket"], accept=True)["kind"] == "ERROR" and not business.orders


def test_unprocessed_supplement_locks_old_confirmation_until_cancel_or_reprepare(agent):
    session, business = agent
    preview = session.chat("帮我报修，水管漏水")

    class Down:
        def decide(self, *args, **kwargs):
            raise POCError("LLM_UNAVAILABLE", "模型连接失败", 503)

    session.planner = Down()
    assert session.chat("不花钱才让师傅来")["kind"] == "ERROR"
    assert session.pending["blocked_reason"] and not business.orders
    assert session.confirm(preview["ticket"], accept=True)["data"]["error"] == "CONDITION_UNRESOLVED"
    assert session.chat("取消本次报修")["kind"] == "CANCELLED"
    assert session.pending is None and not business.orders


def test_v3_elevator_reply_never_invents_coordination_or_repair_deadline(agent):
    session, business = agent
    reply = session.chat("你们这物业真的烦死了，电梯又坏了，今天能修好吗？")
    assert "现场确认" in reply["text"]
    assert all(t not in reply["text"] for t in ("已关注", "正在协调", "已安排", "专人", "今天一定", "投诉"))
    assert session.pending is None and not business.orders


def test_v3_multiple_questions_are_preserved_and_repair_method_does_not_create_draft(agent):
    session, business = agent
    session.chat("这周末小区有什么活动？另外我家卫生间漏水要怎么报修？")
    queries = [call[1] for call in business.calls if call[0] == "consult"]
    assert len(queries) == 2 and "活动" in queries[0] and "报修" in queries[1]
    assert session.pending is None and not business.orders


def test_v3_fixed_phone_remains_byte_identical_during_out_of_scope_followups(agent):
    session, business = agent
    for question in (
        "附近有什么好吃的餐厅推荐吗？",
        "你不知道？",
        "天啊好笨",
        "你认识chatgpt吗，它在哪栋楼？",
        "转人工",
    ):
        reply = session.chat(question)
        assert reply["text"] == FIXED_SERVICE_REPLY if question == "转人工" else reply["text"] != FIXED_SERVICE_REPLY
    assert not business.orders


def test_v3_verified_knowledge_and_ordinary_followup_do_not_overwrite_pending_repair(agent):
    session, business = agent
    p = session.chat("帮我报修，水管漏水")
    reply = session.chat("我们小区允许装修的时间是几点到几点？周末可以装修吗？")
    assert reply["kind"] == "KNOWLEDGE" and "answer_text" in reply["data"]
    assert session.pending["ticket"] == p["ticket"] and not business.orders
    session.set_context(session.unit_id, "current")
    assert not session.dialogue_history and not session.repair_dialogue and not session.repair_details


def completion(decision):
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "plan_property_turn",
                                "arguments": json.dumps(decision, ensure_ascii=False),
                            },
                        }
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }


def test_actual_llm_adapter_is_used_in_workflow_with_history_but_no_credentials(tmp_path):
    business = FakeBusiness(tmp_path)
    requests = []
    scripted = [
        {
            "route": "SERVICE",
            "tasks": [{"intent": "REPAIR_GUIDANCE", "source_quote": "厨房水管漏水怎么办？"}],
            "repair": {"location": "厨房", "symptom": "厨房水管漏水"},
        },
        {
            "route": "SERVICE",
            "tasks": [{"intent": "REPAIR_PREPARE", "source_quote": "帮我报修，今晚七点后有人在家"}],
            "repair": {"location": "厨房", "available_time": "今晚七点后"},
            "use_repair_context": True,
        },
    ]

    def transport(payload):
        requests.append(copy.deepcopy(payload))
        return completion(scripted.pop(0))

    settings = LLMSettings(mode="online", api_key="test-secret-never-sent-in-json")
    planner = LLMPlanner(settings, transport=transport, enable_natural_reply=False)
    session = AgentSession(business, business.token(), planner=planner)
    assert "现场检查" in session.chat("厨房水管漏水怎么办？")["text"]
    p = session.chat("帮我报修，今晚七点后有人在家")
    assert p["kind"] == "REPAIR_PREVIEW" and "厨房水管漏水" in p["data"]["description"]
    assert "厨房水管漏水" in requests[1]["messages"][1]["content"]
    for payload in requests:
        assert "parallel_tool_calls" not in payload, "不要向DeepSeek发送其接口文档未声明的参数"
        assert session.token not in str(payload) and "test-secret-never-sent-in-json" not in str(payload)
        assert session.unit_id not in str(payload) and "resident_id" not in str(payload)
    assert session.confirm(p["ticket"], accept=True)["kind"] == "ORDER_CREATED"
    assert len(business.orders) == 1 and planner.last_metrics["valid_schema"]
    session.close()


@pytest.mark.parametrize(
    "mutation",
    ["tool_name", "identity", "policy_answer", "truncated", "multi_call", "invented_quote", "invented_slot"],
)
def test_invalid_model_outputs_never_execute_actions(tmp_path, mutation):
    business = FakeBusiness(tmp_path)
    reply = completion(
        {"route": "SERVICE", "tasks": [{"intent": "REPAIR_PREPARE", "source_quote": "帮我报修，水管漏水"}]}
    )
    call = reply["choices"][0]["message"]["tool_calls"][0]
    args = json.loads(call["function"]["arguments"])
    if mutation == "tool_name":
        call["function"]["name"] = "execute_cypher"
    if mutation == "identity":
        args["estate_id"] = "E002"
    if mutation == "policy_answer":
        args["answer_text"] = "维修免费，已经安排"
    if mutation == "truncated":
        reply["choices"][0]["finish_reason"] = "length"
    if mutation == "multi_call":
        reply["choices"][0]["message"]["tool_calls"].append(copy.deepcopy(call))
    if mutation == "invented_quote":
        args["tasks"][0]["source_quote"] = "另一个用户要报修"
    if mutation == "invented_slot":
        args["repair"] = {"available_time": "明天下午六点"}
    call["function"]["arguments"] = json.dumps(args)
    planner = LLMPlanner(LLMSettings(mode="online", api_key="test"), transport=lambda _: reply)
    session = AgentSession(business, business.token(), planner=planner)
    assert session.chat("帮我报修，水管漏水")["kind"] == "ERROR"
    assert not business.calls and not business.orders and session.pending is None
    session.close()


def test_schema_disallows_free_prose_and_all_object_fields_are_explicit():
    schema = strict_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert "answer_text" not in schema["properties"]


@pytest.mark.parametrize(
    "error",
    [
        401,
        429,
        httpx.ConnectError("secret-error-body"),
        TimeoutError("secret-error-body"),
    ],
)
def test_http_failure_is_safe_and_does_not_expose_key_or_provider_error(error):
    def send(request):
        if isinstance(error, int):
            return httpx.Response(error, text="secret-error-body")
        raise error
    planner = LLMPlanner(LLMSettings(mode="online", api_key="local-key"))
    planner._client = httpx.Client(transport=httpx.MockTransport(send))
    with pytest.raises(POCError) as info:
        planner._http({})
    assert "secret-error-body" not in info.value.message and "local-key" not in info.value.message
    planner.close()


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_redirect_never_forwards_authorization_to_another_host(status):
    requests = []
    def send(request):
        requests.append(request)
        return httpx.Response(status, headers={"Location": "https://other-host"})
    planner = LLMPlanner(LLMSettings(mode="online", api_key="local-key"))
    planner._client = httpx.Client(transport=httpx.MockTransport(send), follow_redirects=False)
    with pytest.raises(POCError):
        planner._http({})
    assert len(requests) == 1 and requests[0].url.host == "api.deepseek.com"
    planner.close()


def test_missing_and_malformed_configuration_never_silently_switch_online_to_rules(tmp_path, monkeypatch):
    assert LLMSettings.load(tmp_path).mode == "local"
    (tmp_path / "config").mkdir()
    (tmp_path / "config/llm.local.json").write_text('{"mode":"online","api_key":""}')
    with pytest.raises(POCError, match="模型配置"):
        LLMSettings.load(tmp_path)
    (tmp_path / "config/llm.local.json").write_text("broken json")
    with pytest.raises(POCError):
        LLMSettings.load(tmp_path)
    monkeypatch.setattr(LLMSettings, "load", lambda: LLMSettings(mode="online", api_key="test"))
    assert isinstance(build_planner(), LLMPlanner)


def test_sensitive_contact_fields_are_redacted_before_transport():
    seen = []

    def transport(payload):
        seen.append(payload)
        return completion({"route": "CLARIFY"})

    LLMPlanner(LLMSettings(mode="online", api_key="test"), transport=transport).decide(
        "请联系13812345678，邮箱owner@example.com", context={"history": []}
    )
    assert "13812345678" not in str(seen) and "owner@example.com" not in str(seen)
