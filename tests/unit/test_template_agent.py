"""Keep the existing multi-turn graph and actions when answer content comes from templates."""

import copy
import json

import pytest
from test_day3_agent import FakeBusiness
from test_steward_conversation import ScriptedPlanner, factual_reply, plan, text_reply

from property_ontology.agent import AgentSession
from property_ontology.agent.llm import LLMPlanner, LLMSettings
from property_ontology.qa.templates import SemanticDecision


def knowledge(message):
    return {"route": "SERVICE", "tasks": [{"intent": "KNOWLEDGE", "source_quote": message,
            "query": "物业费是多少", "service_code": "property_fee"}]}


def install_answer(business, source, *, status="ANSWERED"):
    original = business.answer_question
    def answer(*args, **kwargs):
        r = original(*args, **kwargs)
        r.update(status=status, answer_source=source, matched_template_id="E001:FAQ:test" if source == "TEMPLATE" else None,
                 match_score=.98 if source == "TEMPLATE" else None, review_status="APPROVED_FOR_DEMO",
                 decision_reason="TEST_ONLY")
        r["debug_info"]["generation_sufficiency"] = "SUFFICIENT" if status == "ANSWERED" else "NONE"
        if status != "ANSWERED":
            r["answer_text"] = "这一点还没有可靠信息。"
        return r
    business.answer_question = answer


def test_template_skips_generator_preserves_draft_confirmation_and_tracking(tmp_path):
    b = FakeBusiness(tmp_path)
    repair, question = "帮我报修，厨房漏水", "物业费是多少"
    planner = ScriptedPlanner([
        (plan(repair, "REPAIR_PREPARE", action_request="EXPLICIT", repair={"location": "厨房", "symptom": "漏水"}), text_reply("核对卡片后再提交。")),
        (knowledge(question), lambda _: pytest.fail("template must not call response model")),
        (plan("查询我的工单进度", "TRACK_ORDER"), factual_reply("")),
    ])
    install_answer(b, "TEMPLATE")
    s = AgentSession(b, b.token(), planner=planner)
    preview = s.chat(repair)
    thread, draft = s.thread_id, copy.deepcopy(s.pending)
    r = s.chat(question)
    assert r["text"] == "只作自动化测试参考" and r["data"]["answer_source"] == "TEMPLATE"
    assert r["data"]["debug_info"]["natural_response"]["generation_mode"] == "STANDARD_ANSWER"
    assert s.pending == draft and s.thread_id == thread and not b.orders
    created = s.confirm(preview["ticket"], accept=True)
    assert created["kind"] == "ORDER_CREATED" and len(b.orders) == 1
    assert s.confirm(preview["ticket"], accept=True)["data"]["replayed"]
    tracked = s.chat("查询我的工单进度")
    assert created["data"]["number"] in tracked["text"] and ("track", created["data"]["number"]) in b.calls
    s.close()


@pytest.mark.parametrize("source,status", [("KNOWLEDGE_GENERATION", "ANSWERED"), ("GENERAL_CONVERSATION", "NOT_FOUND")])
def test_non_template_uses_natural_grounded_generation(tmp_path, source, status):
    b, q = FakeBusiness(tmp_path), "具体怎么回事"
    install_answer(b, source, status=status)
    def response(context):
        return {"segments": [{"kind": "FACT", "fact_index": f["index"]} for f in context["facts"]]
                or [{"kind": "TEXT", "text": "这个细节还没核实，我不能直接确定。"}], "addressed_tasks": [0]}
    planner = ScriptedPlanner([(knowledge(q), response)])
    s = AgentSession(b, b.token(), planner=planner)
    r = s.chat(q)
    assert hasattr(planner, "response_context") and r["data"]["answer_source"] == source
    assert r["data"]["debug_info"]["natural_response"]["generation_mode"] == "LLM_GROUNDED"
    assert not b.orders
    s.close()


def test_semantic_wire_contract_includes_aliases_context_and_no_secrets(tmp_path):
    requests = []
    decision = SemanticDecision(matches=[], sufficiency="NONE")
    def transport(payload):
        requests.append(payload)
        return {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [{"function": {
            "name": "match_property_templates", "arguments": decision.model_dump_json()}}]}}]}
    p = LLMPlanner(LLMSettings(mode="online", api_key="sk-test-never-send"), transport=transport)
    template = {"template_id": "E001:FAQ:test", "version_hash": "v", "canonical_question": "服务电话",
        "question_variants": ["中控室电话"], "standard_answer": "0532-12345678", "conditions": {}, "priority": 100}
    p.match_templates("那晚上呢", templates=[template], knowledge=[], context={
        "conversation_history": [{"role": "user", "content": "中控室联系电话"}], "verified_conditions": {"residence_category": "RESIDENTIAL"}})
    sent = json.loads(requests[0]["messages"][1]["content"])
    assert sent["templates"][0]["question_variants"] == ["中控室电话"]
    assert sent["context"]["conversation_history"] and "sk-test" not in str(requests)
    assert requests[0]["tool_choice"]["function"]["name"] == "match_property_templates"


def test_model_change_disables_old_calibration(tmp_path, monkeypatch):
    import property_ontology.agent.llm as llm
    (tmp_path / "config").mkdir()
    (tmp_path / "config/template_matching.json").write_text(json.dumps({"threshold": .91, "margin": .05, "calibration_model": "old"}))
    monkeypatch.setattr(llm, "ROOT", tmp_path)
    p = LLMPlanner(LLMSettings(mode="online", model="new", api_key="test-only"), transport=lambda _: None)
    assert p.match_settings.threshold is None and p.match_settings.margin is None


def test_wire_short_references_restore_immutable_ids_and_versions():
    def transport(payload):
        sent = json.loads(payload["messages"][1]["content"])
        assert sent["templates"][0]["template_id"] == "T0" and sent["templates"][0]["version_hash"] == "V0"
        assert sent["knowledge"][0]["id"] == "K0" and "unit_id" not in sent["context"]
        return {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [{"function": {
            "name": "match_property_templates", "arguments": json.dumps({"matches": [{"template_id": "T0", "version_hash": "V0",
                "coverage": "FULL", "score": .97, "reason": "direct"}], "knowledge_ids": ["K0"]})}}]}}]}
    p = LLMPlanner(LLMSettings(mode="online", api_key="test-only"), transport=transport)
    t = {"template_id": "E001:FAQ:real", "version_hash": "immutable-source-hash", "canonical_question": "标准问题",
        "question_variants": [], "standard_answer": "标准答案", "conditions": {}, "priority": 100}
    r = p.match_templates("问题", templates=[t], knowledge=[{"id": "E001:POL:real", "text": "事实"}], context={"unit_id": "internal"})
    assert r.matches[0].template_id == t["template_id"] and r.matches[0].version_hash == t["version_hash"]
    assert r.knowledge_ids == ["E001:POL:real"]


def test_wire_reference_outside_the_offered_list_is_rejected_with_internal_diagnostics():
    from property_ontology.domain import POCError
    def transport(payload):
        return {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [{"function": {
            "name": "match_property_templates", "arguments": '{"knowledge_ids":["K999"],"sufficiency":"SUFFICIENT"}'}}]}}]}
    p = LLMPlanner(LLMSettings(mode="online", api_key="test-only"), transport=transport)
    with pytest.raises(POCError) as captured:
        p.match_templates("问题", templates=[], knowledge=[], context={})
    assert captured.value.diagnostics["stage"] == "SEMANTIC_WIRE_REFERENCE"


def test_multi_need_template_preserved_while_repair_waits_for_confirmation(tmp_path):
    b = FakeBusiness(tmp_path)
    install_answer(b, "TEMPLATE")
    q = "物业费是多少？另外厨房漏水，帮我报修"
    decision = knowledge("物业费是多少")
    decision["tasks"].append({"intent": "REPAIR_PREPARE", "source_quote": "厨房漏水，帮我报修"})
    decision.update(action_request="EXPLICIT", repair={"location": "厨房", "symptom": "漏水"})
    planner = ScriptedPlanner([(decision, factual_reply(""))])
    s = AgentSession(b, b.token(), planner=planner)
    reply = s.chat(q)
    assert reply["kind"] == "REPAIR_PREVIEW" and s.pending and not b.orders
    assert reply["text"].count("只作自动化测试参考") == 1
    assert "确认" in reply["text"] and len(planner.response_context["outcomes"]) == 2
    s.close()


def test_online_out_of_scope_cannot_claim_no_knowledge_without_a_controlled_lookup(tmp_path):
    b, q = FakeBusiness(tmp_path), "某项服务的联系信息"
    install_answer(b, "TEMPLATE")
    planner = ScriptedPlanner([({"route": "OUT_OF_SCOPE"}, lambda _: pytest.fail("standard must skip generation"))])
    planner.match_templates = lambda *args, **kwargs: None
    s = AgentSession(b, b.token(), planner=planner)
    r = s.chat(q)
    assert r["data"]["answer_source"] == "TEMPLATE" and r["data"]["debug_info"]["pre_fallback_lookup"]["performed"]
    assert any(c[0] == "consult" for c in b.calls) and not b.orders
    s.close()


def test_online_limits_intent_also_looks_up_knowledge_before_denying_it(tmp_path):
    b, q = FakeBusiness(tmp_path), "服务联系信息"
    install_answer(b, "TEMPLATE")
    planner = ScriptedPlanner([({"route": "SERVICE", "tasks": [{"intent": "EXPLAIN_LIMITS", "source_quote": q}]},
        lambda _: pytest.fail("must return standard directly"))])
    planner.match_templates = lambda *args, **kwargs: None
    s = AgentSession(b, b.token(), planner=planner)
    r = s.chat(q)
    assert r["data"]["answer_source"] == "TEMPLATE" and any(c[0] == "consult" for c in b.calls)
    s.close()


def test_boundary_route_without_user_evidence_is_a_protocol_error():
    from property_ontology.domain import POCError
    def transport(payload):
        return {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [{"function": {
            "name": "plan_property_turn", "arguments": '{"route":"RED_LINE"}'}}]}}]}
    p = LLMPlanner(LLMSettings(mode="online", api_key="test-only"), transport=transport)
    with pytest.raises(POCError) as error:
        p.decide("服务号码是多少", context={})
    assert error.value.code == "INVALID_TOOL_PLAN"
