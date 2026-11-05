"""Stable dialogue state, grounded generation and recovery: model mocks test boundaries, not language quality."""

import copy

import pytest
from test_day3_agent import FakeBusiness

from property_ontology.agent import AgentSession
from property_ontology.agent.dialogue_contract import DialogueDecision
from property_ontology.agent.reply_generation import ReplyDraft, render_grounded
from property_ontology.domain import POCError


class ScriptedPlanner:
    enable_natural_reply = True
    last_response_metrics = {"model": "SCRIPTED_TEST_ONLY", "usage": {}}

    def __init__(self, turns):
        self.turns = iter(turns)
        self.contexts = []

    def decide(self, message, *, context):
        self.contexts.append(copy.deepcopy(context))
        self.turn = next(self.turns)
        return DialogueDecision.model_validate(self.turn[0])

    def respond(self, message, *, context):
        self.response_context = copy.deepcopy(context)
        value = self.turn[1]
        return ReplyDraft.model_validate(value(context) if callable(value) else value)


def text_reply(text, *, task_count=1, question_slot="NONE"):
    return {"segments": [{"kind": "TEXT", "text": text}], "addressed_tasks": list(range(task_count)), "question_slot": question_slot}


def plan(message, intent="REPAIR_GUIDANCE", **extra):
    return {"route": "SERVICE", "tasks": [{"intent": intent, "source_quote": message}], **extra}


def test_floor_answer_is_remembered_with_assistant_question_and_never_asked_again(tmp_path):
    business = FakeBusiness(tmp_path)
    first = "厨房漏水怎么办？"
    planner = ScriptedPlanner([
        (plan(first, repair={"location": "厨房", "symptom": "漏水"}), text_reply("先留意积水附近的电器，不要触碰。水是从哪一处渗出的？", question_slot="part")),
        (plan("地面", repair={"part": "地面"}, use_repair_context=True), text_reply("明白，是厨房地面有水。原因还需要现场查看；现在方便的话，可以先安全关闭供水阀门。")),
        (plan("为什么？", "REPAIR_CAUSE", use_repair_context=True), text_reply("还没经过检查，我不能直接判断是哪处管件出了问题。厨房地面有水只是现象，还需要现场确认来源。")),
    ])
    session = AgentSession(business, business.token(), planner=planner)
    thread = session.thread_id
    for message in (first, "地面", "为什么？"):
        assert session.chat(message)["data"]["debug_info"]["natural_response"]["generation_mode"] == "LLM_GROUNDED"
    history = session.conversation_state["conversation_history"]
    assert [h["role"] for h in history] == ["user", "assistant"] * 3
    assert planner.contexts[1]["conversation_history"][-1]["content"].endswith("水是从哪一处渗出的？")
    assert session.repair_details["location"] == "厨房" and session.repair_details["part"] == "地面"
    assert session.thread_id == thread
    graph = session._conversation_graph.get_state({"configurable": {"thread_id": thread}})
    assert graph.values["conversation_history"] == history and graph.values["collected_slots"]["part"] == "地面"
    assert not business.orders
    session.close()


def test_cancel_and_resume_keep_slots_but_old_confirmation_is_unusable(tmp_path):
    business = FakeBusiness(tmp_path)
    first = "帮我报修，厨房漏水，今晚七点后有人在家"
    planner = ScriptedPlanner([
        (plan(first, "REPAIR_PREPARE", action_request="EXPLICIT", repair={"location": "厨房", "symptom": "漏水", "available_time": "今晚七点后"}), text_reply("时间意向记住了，请核对卡片，确认后再提交。")),
        ({"route": "CANCEL", "context_operation": "CANCEL"}, text_reply("好，先不报修，前面说的情况我还记着。")),
        (plan("那还是帮我报修吧", "REPAIR_PREPARE", context_operation="RESUME", action_request="EXPLICIT"), text_reply("我们接着刚才的漏水问题处理，重新核对卡片后再提交。")),
    ])
    session = AgentSession(business, business.token(), planner=planner)
    old = session.chat(first)
    assert old["kind"] == "REPAIR_PREVIEW"
    assert session.chat("先不报修了")["kind"] == "CANCELLED"
    assert session.pending is None and session.repair_details["available_time"] == "今晚七点后"
    assert session.confirm(old["ticket"], accept=True)["kind"] == "ERROR"
    new = session.chat("那还是帮我报修吧")
    assert new["ticket"] != old["ticket"] and "厨房" in new["data"]["description"] and "今晚七点后" in new["data"]["description"]
    created = session.confirm(new["ticket"], accept=True)
    assert len(business.orders) == 1 and created["kind"] == "ORDER_CREATED"
    state = session._conversation_graph.get_state({"configurable": {"thread_id": session.thread_id}}).values
    assert state["last_tool_result"]["number"] == created["data"]["number"]
    assert state["active_task"]["status"] == "SUBMITTED"
    session.close()


def test_generic_precondition_cannot_be_satisfied_by_model_or_click(tmp_path):
    business = FakeBusiness(tmp_path)
    message = "预算不超过五十元才帮我报修"
    planner = ScriptedPlanner([
        (plan(message, "REPAIR_PREPARE", action_request="CONDITIONAL", condition_quote="预算不超过五十元"), text_reply("费用还没核实，先不提交，不能替你认定预算条件已经满足。")),
        (plan("那就确认", "REPAIR_PREPARE", action_request="EXPLICIT"), text_reply("费用前提还没核实，确认这句话也不能代替满足预算条件。")),
    ])
    session = AgentSession(business, business.token(), planner=planner)
    session.chat(message)
    session.chat("那就确认")
    assert session.pending is None and not business.orders
    assert session.conversation_state["pending_conditions"][0]["status"] == "UNRESOLVED"
    session.close()


def test_correction_updates_only_requested_slot_and_stops_old_draft(tmp_path):
    business = FakeBusiness(tmp_path)
    first = "帮我报修，厨房漏水，今晚七点后有人在家"
    planner = ScriptedPlanner([
        (plan(first, "REPAIR_PREPARE", action_request="EXPLICIT", repair={"location": "厨房", "symptom": "漏水", "available_time": "今晚七点后"}), text_reply("请先核对报修内容，再确认提交。")),
        ({"route": "CORRECTION", "context_operation": "CORRECT", "repair": {"location": "卫生间"}}, text_reply("明白，是卫生间，之前说的时间仍保留。旧卡片先停下，免得按错的位置提交。")),
    ])
    session = AgentSession(business, business.token(), planner=planner)
    preview = session.chat(first)
    answer = session.chat("不是厨房，是卫生间")
    assert answer["kind"] == "CORRECTED" and session.pending is None
    assert session.repair_details == {"location": "卫生间", "symptom": "漏水", "available_time": "今晚七点后"}
    assert session.confirm(preview["ticket"], accept=True)["kind"] == "ERROR" and not business.orders
    session.close()


def minimal_context(**update):
    return {"required_fact_indices": [], "outcomes": [{"task_index": 0}], "collected_slots": {},
            "pending_conditions": [], "facts": [], **update}


@pytest.mark.parametrize("text", ["物业费是9.99元", "已经安排师傅了", "正在协调维修公司", "会安排师傅上门", "这通常是免费的", "他的住址是3栋", "来源是data/raw/a.docx"])
def test_ungrounded_generation_is_rejected(text):
    with pytest.raises(POCError):
        render_grounded(text_reply(text), minimal_context())


def test_policy_fact_is_inserted_once_without_rewriting_and_multi_task_cannot_be_dropped():
    context = minimal_context(required_fact_indices=[0], facts=[{"text": "住宅物业费为2.16元/平方米/月。"}],
                              outcomes=[{"task_index": 0}, {"task_index": 1}])
    draft = {"segments": [{"kind": "FACT", "fact_index": 0}, {"kind": "TEXT", "text": "活动安排目前还没有可靠信息。"}], "addressed_tasks": [0, 1]}
    assert render_grounded(draft, context).count("2.16") == 1
    with pytest.raises(POCError):
        render_grounded({**draft, "addressed_tasks": [0]}, context)
    with pytest.raises(POCError):
        render_grounded({**draft, "segments": [draft["segments"][0]] * 2}, context)
    with pytest.raises(POCError):
        render_grounded(text_reply("在哪里漏水？", question_slot="part"), minimal_context(collected_slots={"part": "地面"}))


def test_composition_failure_preserves_prepared_state_and_never_claims_order_created(tmp_path):
    business = FakeBusiness(tmp_path)
    message = "帮我报修，厨房漏水"
    planner = ScriptedPlanner([(plan(message, "REPAIR_PREPARE", action_request="EXPLICIT", repair={"location": "厨房", "symptom": "漏水"}), text_reply("已经安排师傅上门了"))])
    session = AgentSession(business, business.token(), planner=planner)
    reply = session.chat(message)
    assert reply["kind"] == "REPAIR_PREVIEW" and session.pending and not business.orders
    assert reply["data"]["debug_info"]["natural_response"]["generation_mode"] == "SAFE_FALLBACK"
    assert "已经安排" not in reply["text"]
    session.close()


def test_say_thanks_or_feedback_does_not_cancel_active_repair(tmp_path):
    business = FakeBusiness(tmp_path)
    first = "帮我报修，厨房漏水"
    planner = ScriptedPlanner([
        (plan(first, "REPAIR_PREPARE", action_request="EXPLICIT", repair={"location": "厨房", "symptom": "漏水"}), text_reply("内容准备好了，请核对卡片后确认。")),
        (plan("真的好笨", "FEEDBACK"), text_reply("刚才确实没接好你的话。我们接着厨房漏水这件事说。")),
        (plan("谢谢", "SOCIAL"), text_reply("不客气，有需要接着说就行。")),
    ])
    session = AgentSession(business, business.token(), planner=planner)
    preview = session.chat(first)
    for message in ("真的好笨", "谢谢"):
        assert session.chat(message)["data"]["debug_info"]["natural_response"]["generation_mode"] == "LLM_GROUNDED"
        assert session.pending["ticket"] == preview["ticket"]
    assert not business.orders
    session.set_context(session.unit_id, "current")
    assert not session.conversation_state["conversation_history"] and not session.conversation_state["collected_slots"]
    session.close()


def test_missing_fault_is_collected_before_preparation_and_only_user_facts_enter_draft(tmp_path):
    business = FakeBusiness(tmp_path)
    planner = ScriptedPlanner([
        (plan("帮我报修", "REPAIR_PREPARE", action_request="EXPLICIT", missing_slot="symptom"),
         text_reply("哪里出了什么问题？", question_slot="symptom")),
        (plan("厨房水龙头滴水", repair={"location": "厨房", "symptom": "水龙头滴水"}),
         text_reply("报修内容准备好了，请核对后确认提交。")),
    ])
    session = AgentSession(business, business.token(), planner=planner)
    assert session.chat("帮我报修")["kind"] == "DIALOGUE" and not session.pending
    reply = session.chat("厨房水龙头滴水")
    assert reply["kind"] == "REPAIR_PREVIEW" and not business.orders
    assert reply["data"]["description"] == "位置：厨房\n现象：水龙头滴水"
    assert "帮我报修" not in reply["data"]["description"]
    session.close()


def test_history_and_checkpoint_storage_remain_bounded_with_stable_thread(tmp_path):
    business = FakeBusiness(tmp_path)
    planner = ScriptedPlanner([(plan("谢谢", "SOCIAL"), text_reply("不客气。"))] * 30)
    session = AgentSession(business, business.token(), planner=planner)
    thread = session.thread_id
    for _ in range(30):
        session.chat("谢谢")
    assert session.thread_id == thread
    assert len(session.conversation_state["conversation_history"]) == 24
    assert len(list(session._conversation_memory.list({"configurable": {"thread_id": thread}}))) == 1
    session.close()
    assert not list(session._conversation_memory.list({"configurable": {"thread_id": thread}}))


def test_response_adapter_rejects_truncated_json_and_uses_non_nullable_question_enum():
    import json

    from property_ontology.agent.llm import LLMPlanner, LLMSettings

    requests = []
    def broken(payload):
        requests.append(payload)
        return {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [{"function": {
            "name": "compose_property_reply", "arguments": '{"segments": [], "question_slot"'}}]}}]}
    planner = LLMPlanner(LLMSettings(mode="online", api_key="test-only"), response_transport=broken)
    with pytest.raises(POCError, match="可靠"):
        planner.respond("地面", context=minimal_context())
    assert planner.last_response_error["errors"][0]["type"] == "json_invalid"
    schema = requests[0]["tools"][0]["function"]["parameters"]
    assert schema["properties"]["question_slot"]["type"] == "string"
    assert "NONE" in schema["properties"]["question_slot"]["enum"]
    assert "test-only" not in json.dumps(requests)


def test_verified_slots_remain_valid_after_original_turn_leaves_recent_history(tmp_path):
    business = FakeBusiness(tmp_path)
    first = "厨房漏水，帮我报修，今晚七点后有人在家"
    planner = ScriptedPlanner([
        (plan(first, "REPAIR_PREPARE", action_request="EXPLICIT", repair={"location": "厨房", "symptom": "漏水", "available_time": "今晚七点后"}), text_reply("请核对后确认。")),
        *[(plan("谢谢", "SOCIAL"), text_reply("不客气。"))] * 15,
        (plan("你不知道？", "FEEDBACK", repair={"location": "厨房", "symptom": "漏水", "available_time": "今晚七点后"}), text_reply("漏水原因还没检查，我不能猜，但前面说的时间意向还记着。")),
    ])
    session = AgentSession(business, business.token(), planner=planner)
    session.chat(first)
    for _ in range(15):
        session.chat("谢谢")
    reply = session.chat("你不知道？")
    assert reply["kind"] != "ERROR" and session.repair_details["available_time"] == "今晚七点后"
    assert not business.orders
    session.close()


def test_uncertainty_is_allowed_but_future_dispatch_and_unneeded_questions_are_rejected():
    assert render_grounded(text_reply("我无法确认是不是免费，先不提交。"), minimal_context())
    for value in ("我帮您安排维修师傅上门检查。", "确认后提交派单。"):
        with pytest.raises(POCError):
            render_grounded(text_reply(value), minimal_context())
    with pytest.raises(POCError, match="无需询问"):
        render_grounded(text_reply("什么时候有人在家？", question_slot="available_time"), minimal_context())
    with pytest.raises(POCError, match="可信按钮"):
        render_grounded(text_reply("您回一句确认提交，我就马上提交。"), minimal_context())
    with pytest.raises(POCError, match="代替用户"):
        render_grounded(text_reply("我先按不管费用都报修来处理。"), minimal_context(
            pending_conditions=[{"text": "如果免费才报修", "status": "UNRESOLVED"}]))


def factual_reply(text):
    def make(context):
        return {"segments": [{"kind": "TEXT", "text": text},
                             *[{"kind": "FACT", "fact_index": i} for i in context["required_fact_indices"]]],
                "addressed_tasks": [o["task_index"] for o in context["outcomes"]]}
    return make


def test_response_repair_does_not_reexecute_preparation(tmp_path):
    business = FakeBusiness(tmp_path)
    attempts = []
    def response(context):
        attempts.append(context.copy())
        return text_reply("已经安排师傅上门了") if len(attempts) == 1 else factual_reply("内容准备好了。")(context)
    message = "帮我报修，厨房漏水"
    planner = ScriptedPlanner([(plan(message, "REPAIR_PREPARE", action_request="EXPLICIT", repair={"symptom": "漏水"}), response)])
    session = AgentSession(business, business.token(), planner=planner)
    reply = session.chat(message)
    assert reply["data"]["debug_info"]["natural_response"]["generation_mode"] == "LLM_GROUNDED"
    assert len(attempts) == 2 and len([c for c in business.calls if c[0] == "prepare"]) == 1
    assert not business.orders and session.pending
    session.close()


def test_social_topic_preserves_business_state_without_injecting_repair_into_wording(tmp_path):
    business = FakeBusiness(tmp_path)
    message = "帮我报修，厨房漏水"
    planner = ScriptedPlanner([
        (plan(message, "REPAIR_PREPARE", action_request="EXPLICIT", repair={"symptom": "漏水"}), factual_reply("内容准备好了。")),
        (plan("你认识ChatGPT吗？", "SOCIAL"), text_reply("认识，它是一个AI对话助手。")),
        (plan("你不知道？", "FEEDBACK"), text_reply("它是AI助手，没有住址，不能当成小区住户。")),
    ])
    session = AgentSession(business, business.token(), planner=planner)
    old = session.chat(message)
    session.chat("你认识ChatGPT吗？")
    assert planner.response_context["collected_slots"] == {} and not planner.response_context["active_task"]
    assert session.repair_details["symptom"] == "漏水" and session.pending["ticket"] == old["ticket"]
    session.chat("你不知道？")
    assert planner.response_context["collected_slots"] == {} and session.conversation_state["current_topic"] == "SOCIAL"
    assert len(session.conversation_state["conversation_history"]) == 6
    session.close()


def test_one_tool_failure_does_not_erase_other_task_results(tmp_path):
    business = FakeBusiness(tmp_path)
    message = "物业费是多少？另外查询工单E001-TEST-99999999"
    decision = {"route": "SERVICE", "tasks": [
        {"intent": "KNOWLEDGE", "source_quote": "物业费是多少", "service_code": "property_fee"},
        {"intent": "TRACK_ORDER", "source_quote": "查询工单E001-TEST-99999999"}]}
    planner = ScriptedPlanner([(decision, factual_reply("工单目前无法查询，可能不存在或当前账号无权查看。"))])
    session = AgentSession(business, business.token(), planner=planner)
    reply = session.chat(message)
    assert "只作自动化测试参考" in reply["text"] and "工单目前无法查询" in reply["text"]
    assert [o["status"] for o in reply["data"]["debug_info"]["dialogue_outcomes"]] == ["ANSWERED", "NOT_AVAILABLE"]
    assert reply["data"]["debug_info"]["raw_matches"]
    assert not business.orders
    session.close()
