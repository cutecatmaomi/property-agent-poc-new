"""Run the conversation graph with bad model plans and real confirmation state."""

import pytest

from test_day3_agent import FakeBusiness
from test_steward_conversation import ScriptedPlanner, plan, text_reply

from property_ontology.agent import AgentSession
from property_ontology.agent.reply_generation import render_grounded
from property_ontology.domain import POCError


CASE4 = [
    "你们这物业真的烦死了，电梯又坏了，今天能修好吗？",
    "你已经联系电梯公司了吗？",
    "那你说的处理进度有依据吗？有工单编号吗？",
]


@pytest.mark.parametrize("estate", ["E001", "E002"])
def test_original_case4_does_not_create_a_draft_even_if_model_labels_it_explicit(tmp_path, estate):
    business = FakeBusiness(tmp_path)
    planner = ScriptedPlanner([
        (plan(CASE4[0], "REPAIR_PREPARE", action_request="EXPLICIT", repair_request_quote=CASE4[0],
              repair={"location": "电梯", "symptom": "电梯又坏了"}),
         text_reply("电梯又出故障确实影响出行。今天能否修好，还需要检修人员核实；先不要使用故障电梯。")),
        (plan(CASE4[1], "OPERATION_STATUS", tasks=[{"intent": "OPERATION_STATUS", "source_quote": CASE4[1],
                                                  "operation_subject": "EXTERNAL_CONTACT"}]), None),
        (plan(CASE4[2], "OPERATION_STATUS", tasks=[{"intent": "OPERATION_STATUS", "source_quote": CASE4[2],
                                                  "operation_subject": "ORDER_RECEIPT"}]), None),
    ])
    with_session = AgentSession(business, business.token(estate), planner=planner)
    try:
        replies = [with_session.chat(message) for message in CASE4]
        assert all(r["kind"] == "DIALOGUE" for r in replies)
        assert "今天能否修好" in replies[0]["text"]
        assert "没有联系" in replies[1]["text"]
        assert "没有新工单编号" in replies[2]["text"]
        for r in replies[1:]:
            assert r["data"]["debug_info"]["natural_response"]["generation_mode"] == "OPERATION_RECEIPT"
            assert r["data"]["generation_llm_call_count"] == 0
            assert all(t not in r["text"] for t in ("确认提交报修", "不该那么说", "专人", "正在协调"))
        assert with_session.pending is None
        assert not business.calls and not business.orders
        assert len(with_session.conversation_state["conversation_history"]) == 6
        assert planner.contexts[1]["operation_state"]["draft_status"] == "NONE"
    finally:
        with_session.close()


@pytest.mark.parametrize("question", [
    "水管又坏了，什么时候能修好？", "门禁不亮了，今天可以恢复吗？", "电梯维修要多久？",
    "停电了，你们怎么处理？", "厨房漏水是什么原因？",
    "电梯又坏了", "厨房漏水",
])
def test_other_fault_questions_cannot_become_authorization_from_a_model_label(tmp_path, question):
    business = FakeBusiness(tmp_path)
    planner = ScriptedPlanner([(plan(question, "REPAIR_PREPARE", action_request="EXPLICIT",
        repair_request_quote=question, repair={"symptom": question}),
        text_reply("具体情况尚未核实，需要先检查故障。"))])
    session = AgentSession(business, business.token(), planner=planner)
    try:
        assert session.chat(question)["kind"] == "DIALOGUE"
        assert session.pending is None and not business.calls and not business.orders
    finally:
        session.close()


def test_receipt_followup_preserves_authorized_draft_without_repreparing_or_urging_submit(tmp_path):
    business = FakeBusiness(tmp_path)
    first, question = "帮我报修，电梯又坏了", CASE4[2]
    planner = ScriptedPlanner([
        (plan(first, "REPAIR_PREPARE", action_request="EXPLICIT", repair={"symptom": "电梯又坏了"}),
         text_reply("请核对报修内容。")),
        (plan(question, "OPERATION_STATUS", tasks=[{"intent": "OPERATION_STATUS", "source_quote": question,
                                                  "operation_subject": "ORDER_RECEIPT"}]), None),
    ])
    session = AgentSession(business, business.token(), planner=planner)
    try:
        old = session.chat(first)
        reply = session.chat(question)
        assert session.pending["ticket"] == old["ticket"]
        assert "尚未提交" in reply["text"] and "确认提交报修" not in reply["text"]
        assert len([c for c in business.calls if c[0] == "prepare"]) == 1
        created = session.confirm(old["ticket"], accept=True)
        assert created["kind"] == "ORDER_CREATED" and len(business.orders) == 1
        assert session.confirm(old["ticket"], accept=True)["data"]["replayed"]
    finally:
        session.close()


def test_copied_slots_on_a_followup_are_not_a_new_draft_authorization(tmp_path):
    business = FakeBusiness(tmp_path)
    first, question = "帮我报修，厨房漏水", "你已经通知维修人员了吗？"
    planner = ScriptedPlanner([
        (plan(first, "REPAIR_PREPARE", action_request="EXPLICIT", repair={"symptom": "漏水"}), text_reply("请核对报修内容。")),
        (plan(question, "REPAIR_PREPARE", action_request="EXPLICIT", repair={"symptom": "漏水"}),
         text_reply("我还没有通知维修人员。")),
    ])
    session = AgentSession(business, business.token(), planner=planner)
    try:
        old = session.chat(first)
        assert session.chat(question)["kind"] == "DIALOGUE"
        assert session.pending["ticket"] == old["ticket"]
        assert len([c for c in business.calls if c[0] == "prepare"]) == 1
    finally:
        session.close()


def test_semantic_request_quote_preserves_natural_non_keyword_requests(tmp_path):
    business = FakeBusiness(tmp_path)
    first, request = "厨房水管漏水", "那就把刚才这事报一单吧"
    planner = ScriptedPlanner([
        (plan(first, repair={"location": "厨房", "symptom": "水管漏水"}), text_reply("原因需要现场检查。")),
        (plan(request, "REPAIR_PREPARE", action_request="EXPLICIT", repair_request_quote=request),
         text_reply("请核对本次报修。")),
    ])
    session = AgentSession(business, business.token(), planner=planner)
    try:
        session.chat(first)
        preview = session.chat(request)
        assert preview["kind"] == "REPAIR_PREVIEW" and not business.orders
        assert "水管漏水" in preview["data"]["description"]
    finally:
        session.close()


def test_operation_receipt_uses_submitted_number_without_claiming_real_dispatch(tmp_path):
    business = FakeBusiness(tmp_path)
    first, query = "帮我报修，厨房漏水", "有工单编号了吗？"
    planner = ScriptedPlanner([
        (plan(first, "REPAIR_PREPARE", action_request="EXPLICIT", repair={"symptom": "漏水"}), text_reply("请核对本次报修。")),
        (plan(query, "OPERATION_STATUS", tasks=[{"intent": "OPERATION_STATUS", "source_quote": query,
                                               "operation_subject": "ORDER_RECEIPT"}]), None),
        (plan("查一下刚才的报修进度", "TRACK_ORDER"),
         lambda c: {"segments": [{"kind": "FACT", "fact_index": f["index"]} for f in c["facts"]], "addressed_tasks": [0]}),
    ])
    session = AgentSession(business, business.token(), planner=planner)
    try:
        p = session.chat(first)
        created = session.confirm(p["ticket"], accept=True)
        reply = session.chat(query)
        assert created["data"]["number"] in reply["text"]
        assert "没有新工单编号" not in reply["text"] and "仍需查询" in reply["text"]
        assert session.chat("查一下刚才的报修进度")["kind"] != "ERROR"
        assert any(c[0] == "track" for c in business.calls) and len(business.orders) == 1
    finally:
        session.close()


def test_reply_cannot_advertise_nonexistent_confirmation_card():
    with pytest.raises(POCError, match="不存在的确认卡片"):
        render_grounded(text_reply("请在卡片上点击确认提交报修。"), {
            "required_fact_indices": [], "facts": [], "outcomes": [{"task_index": 0}],
            "collected_slots": {}, "pending_conditions": [], "missing_slots": [], "has_pending_confirmation": False})


def test_conditional_offer_is_not_mistaken_for_an_existing_confirmation_card():
    context = {"required_fact_indices": [], "facts": [], "outcomes": [{"task_index": 0}],
               "collected_slots": {}, "pending_conditions": [], "missing_slots": [], "has_pending_confirmation": False}
    offer = "需要的话我可以帮您准备一份报修草稿，您确认后提交。"
    assert render_grounded(text_reply(offer), context) == offer


def test_streamlit_original_case4_rerun_has_one_answer_per_turn_and_no_card(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from property_ontology.agent import ui_backend
    from property_ontology.config import ROOT

    business = FakeBusiness(tmp_path)
    planner = ScriptedPlanner([
        (plan(CASE4[0], "REPAIR_PREPARE", action_request="EXPLICIT", repair={"symptom": "电梯又坏了"}),
         text_reply("今天能否修好还需检修确认，请先别使用故障电梯。")),
        (plan(CASE4[1], "OPERATION_STATUS", tasks=[{"intent": "OPERATION_STATUS", "source_quote": CASE4[1],
                                                  "operation_subject": "EXTERNAL_CONTACT"}]), None),
        (plan(CASE4[2], "OPERATION_STATUS", tasks=[{"intent": "OPERATION_STATUS", "source_quote": CASE4[2],
                                                  "operation_subject": "ORDER_RECEIPT"}]), None),
    ])
    monkeypatch.setattr(ui_backend, "open_session", lambda *args: AgentSession(business, business.token(), planner=planner))
    app = AppTest.from_file(str(ROOT / "src/property_ontology/streamlit_app.py"), default_timeout=20).run()
    app.text_input(key="login_password").input("mock-password")
    app.button[0].click().run()
    for i, message in enumerate(CASE4, 1):
        app.chat_input[0].set_value(message).run()
        assert not app.exception and len(app.chat_message) == i * 2
        assert not any(b.key == "confirm_repair" for b in app.button)
    for _ in range(3):
        app.run()
        assert len(app.chat_message) == 6 and not app.exception
        assert len(planner.contexts) == 3
        assert not business.calls and not business.orders
    app.session_state["agent_session"].close()
