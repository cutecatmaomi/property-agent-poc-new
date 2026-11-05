"""Browser-session behavior through Streamlit's real AppTest runner, with synthetic business storage."""

import pytest
from streamlit.testing.v1 import AppTest
from test_day3_agent import FakeBusiness

from property_ontology.agent import AgentSession, ui_backend
from property_ontology.config import ROOT
from property_ontology.domain import POCError
from property_ontology.response_policy import FIXED_SERVICE_REPLY


@pytest.fixture
def ui(tmp_path, monkeypatch):
    business = FakeBusiness(tmp_path)

    def login(scope, user, password):
        token = business.auth.login(scope, user, password)["token"]
        return AgentSession(business, token)

    monkeypatch.setattr(ui_backend, "open_session", login)
    app = AppTest.from_file(str(ROOT / "src/property_ontology/streamlit_app.py"), default_timeout=20).run()
    assert not app.exception
    return app, business


def login(app, scope="E001", user="owner1"):
    app.selectbox(key="login_estate").select(scope)
    app.selectbox(key="login_username").select(user)
    app.text_input(key="login_password").input("mock-password")
    app.button[0].click().run()
    assert not app.exception and len(app.chat_input) == 1


def test_login_failure_and_query_sources_in_demo_and_current(ui):
    app, _ = ui
    app.text_input(key="login_password").input("wrong")
    app.button[0].click().run()
    assert app.error and not app.chat_input
    login(app)
    app.chat_input[0].set_value("合同什么时候可以拿").run()
    assert not app.exception and "只作自动化测试参考" in "\n".join(t.value for t in app.text)
    assert "Sheet1" not in "\n".join(t.value for t in app.text)
    app.selectbox(key="knowledge_mode").select("current").run()
    assert not app.chat_message
    app.chat_input[0].set_value("合同什么时候可以拿").run()
    assert "暂时无法确认" in "\n".join(t.value for t in app.text)


def test_ui_button_confirmation_repeat_and_order_tracking(ui):
    app, business = ui
    login(app)
    app.chat_input[0].set_value("我要报修，厨房水龙头滴水").run()
    assert not app.exception and not business.orders
    app.chat_input[0].set_value("确认").run()
    assert not business.orders
    app.button(key="confirm_repair").click().run()
    assert not app.exception and len(business.orders) == 1
    first = next(iter(business.orders.values()))["number"]
    app.button(key="repeat_repair").click().run()
    assert not app.exception and len(business.orders) == 1
    assert "重复确认" in "\n".join(t.value for t in app.text)
    app.chat_input[0].set_value("查询我的工单进度").run()
    assert first in "\n".join(t.value for t in app.text)


def test_cancel_and_logout_switch_clear_history_and_confirmation(ui):
    app, business = ui
    login(app)
    old_session = app.session_state["agent_session"]
    app.chat_input[0].set_value("我要报修，灯坏了").run()
    ticket = old_session.pending["ticket"]
    app.button(key="cancel_repair").click().run()
    assert not business.orders and old_session.pending is None
    app.chat_input[0].set_value("我要报修，水管漏水").run()
    app.button(key="logout").click().run()
    assert not app.exception and not app.chat_message and not app.chat_input
    with pytest.raises(POCError):
        business.auth.resolve(old_session.token)
    login(app, scope="E002")
    assert app.session_state["agent_session"].estate_id == "E002" and not app.chat_message
    assert app.session_state["agent_session"].confirm(ticket, accept=True)["kind"] == "ERROR"
    assert not business.orders


def test_fixed_reply_is_exact_and_expired_session_stops_chat(ui):
    app, business = ui
    login(app)
    app.chat_input[0].set_value("转人工").run()
    assert FIXED_SERVICE_REPLY in [t.value for t in app.text]
    assert not business.orders
    session = app.session_state["agent_session"]
    business.auth.logout(session.token)
    app.run()
    assert app.error and not app.chat_input and not app.chat_message


def test_chat_history_never_displays_streamlit_internal_objects(ui):
    app, _ = ui
    login(app)
    app.chat_input[0].set_value("合同什么时候可以拿").run()
    assert not app.exception and len(app.chat_message) == 2
    assert "合同什么时候可以拿" in [t.value for t in app.text]
    assert "只作自动化测试参考" in "\n".join(t.value for t in app.text)
    assert not app.get("help_info"), "聊天内容不能产生DeltaGenerator等内部对象说明"
    assert not app.get("code"), "聊天历史不能额外展示页面代码"
    app.run()
    assert not app.exception and not app.get("help_info") and not app.get("code")


def test_public_chat_uses_only_answer_text_and_reruns_do_not_duplicate_messages(ui):
    app, _ = ui
    login(app)
    app.chat_input[0].set_value("物业费是多少").run()
    for _ in range(3):
        app.run()
        assert len(app.chat_message) == 2 and not app.exception
        assert [t.value for t in app.text].count("只作自动化测试参考") == 1
        rendered = "\n".join(t.value for t in app.text)
        assert all(value not in rendered for value in ("data/raw", "Sheet1", "UNIMPLEMENTED_ACTION", "POC演示资料", "核验原文"))
        assert not app.expander


def test_template_hit_is_rendered_once_without_regeneration_on_rerun(ui, monkeypatch):
    from test_steward_conversation import ScriptedPlanner
    from test_template_agent import install_answer, knowledge

    app, business = ui
    login(app)
    agent = app.session_state["agent_session"]
    install_answer(business, "TEMPLATE")
    clock = iter([10.0, 11.234])
    monkeypatch.setattr(ui_backend, "perf_counter", lambda: next(clock))
    def no_generation(_):
        pytest.fail("模板命中后不应调用自由生成")
    planner = ScriptedPlanner([(knowledge("物业费是多少"), no_generation)])
    agent.planner = planner
    app.chat_input[0].set_value("物业费是多少").run()
    assert not app.exception and len(planner.contexts) == 1
    assert [c.value for c in app.caption].count("本次回答耗时 1.23 秒") == 1
    assert app.session_state["messages"][-1]["reply"]["text"] == "只作自动化测试参考"
    assert all("本次回答耗时" not in m["content"] for m in agent.conversation_state["conversation_history"])
    calls = len(business.calls)
    for _ in range(3):
        app.run()
        assert not app.exception and len(app.chat_message) == 2
        assert [t.value for t in app.text].count("只作自动化测试参考") == 1
        assert len(planner.contexts) == 1 and len(business.calls) == calls
        assert [c.value for c in app.caption].count("本次回答耗时 1.23 秒") == 1
        assert not app.get("code") and not app.get("help_info")


def test_no_loading_spinner_and_user_message_renders_before_backend(ui, monkeypatch):
    import streamlit as st

    app, _ = ui
    login(app)
    events = []
    original_message = st.chat_message
    agent = app.session_state["agent_session"]
    original_chat = agent.chat

    def message(role, *args, **kwargs):
        events.append(role)
        return original_message(role, *args, **kwargs)

    def chat(value):
        assert events[-1] == "user", "发送内容应先显示，随后等待后台结果"
        events.append("backend")
        return original_chat(value)

    def spinner(*args, **kwargs):
        pytest.fail("聊天不应创建加载转圈")

    monkeypatch.setattr(st, "chat_message", message)
    monkeypatch.setattr(st, "spinner", spinner)
    monkeypatch.setattr(agent, "chat", chat)
    app.chat_input[0].set_value("物业费是多少").run()
    assert not app.exception and len(app.chat_message) == 2
    assert not app.get("spinner")
    assert 'data-testid="stStatusWidget"' in app.get("html")[0].proto.body
    assert ':has([data-testid="stStatusWidgetRunningIcon"])' in app.get("html")[0].proto.body
    app.run()
    assert not app.exception and len(app.chat_message) == 2


def test_each_chat_and_confirmation_keeps_its_own_response_time(ui, monkeypatch):
    app, business = ui
    login(app)
    clock = iter([10.0, 11.25, 20.0, 22.5, 30.0, 30.125, 40.0, 40.75, 50.0, 51.0])
    monkeypatch.setattr(ui_backend, "perf_counter", lambda: next(clock))
    app.chat_input[0].set_value("我要报修，厨房水龙头滴水").run()
    app.button(key="confirm_repair").click().run()
    app.button(key="repeat_repair").click().run()
    assert len(business.orders) == 1
    app.chat_input[0].set_value("我要报修，卧室灯坏了").run()
    app.button(key="cancel_repair").click().run()
    assert not app.exception and len(business.orders) == 1
    expected = ["本次回答耗时 1.25 秒", "本次回答耗时 2.50 秒", "本次回答耗时 0.12 秒",
                "本次回答耗时 0.75 秒", "本次回答耗时 1.00 秒"]
    for _ in range(3):
        app.run()
        assert not app.exception
        assert [c.value for c in app.caption if c.value.startswith("本次回答耗时")] == expected
        assert len(app.chat_message) == 7 and len(business.orders) == 1


def test_historical_reply_without_timing_does_not_invent_a_duration(ui):
    app, _ = ui
    login(app)
    app.session_state["messages"] = [{"role": "assistant", "text": "之前的答复",
                                      "reply": {"text": "之前的答复", "data": {}}}]
    app.run()
    assert not app.exception and [t.value for t in app.text].count("之前的答复") == 1
    assert not [c for c in app.caption if c.value.startswith("本次回答耗时")]


def test_natural_dialogue_ui_rerun_preserves_history_without_extra_model_calls(tmp_path, monkeypatch):
    from test_steward_conversation import ScriptedPlanner, plan, text_reply

    business = FakeBusiness(tmp_path)
    planner = ScriptedPlanner([
        (plan("厨房漏水怎么办？", repair={"location": "厨房", "symptom": "漏水"}),
         text_reply("先远离积水旁的电器。水从哪里渗出来？", question_slot="part")),
        (plan("地面", repair={"part": "地面"}), text_reply("明白，是厨房地面渗水，原因还需要现场检查。")),
    ])
    def open_agent(scope, user, password):
        return AgentSession(business, business.auth.login(scope, user, password)["token"], planner=planner)
    monkeypatch.setattr(ui_backend, "open_session", open_agent)
    app = AppTest.from_file(str(ROOT / "src/property_ontology/streamlit_app.py")).run()
    login(app)
    app.chat_input[0].set_value("厨房漏水怎么办？").run()
    app.chat_input[0].set_value("地面").run()
    thread = app.session_state["agent_session"].thread_id
    for _ in range(3):
        app.run()
        assert not app.exception and len(app.chat_message) == 4
        assert len(planner.contexts) == 2
        assert app.session_state["agent_session"].thread_id == thread
        assert [t.value for t in app.text].count("明白，是厨房地面渗水，原因还需要现场检查。") == 1
        assert not app.get("help_info") and not app.get("code")
