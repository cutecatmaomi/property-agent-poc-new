"""Full Agent/Action/selector tests with explicit model stubs (no paid requests)."""
import copy
import json

import pytest
from golden_support import ROWS, BY_CASE, PacketBusiness
from test_llm_dialogue import completion
from test_steward_conversation import plan, text_reply

from property_ontology.agent import AgentSession
from property_ontology.agent.llm import LLMPlanner, LLMSettings
from property_ontology.domain import POCError


def selection(catalog, target, **overrides):
    original = next(r for r in ROWS if r['qa_id'] == target)
    t = next(t for t in catalog if t['canonical_question'] == original['question'])
    return {'matches': [{'template_id': t['template_id'], 'version_hash': t['version_hash'],
        'coverage': 'FULL', 'score': .99, 'reason': 'scripted semantic decision',
        'question_relation': 'EQUIVALENT', 'additional_fact_required': False, **overrides}]}


def model_transport(calls, target, *, damage=None, query=None):
    def send(payload):
        calls.append(payload)
        name = payload['tools'][0]['function']['name']
        context = json.loads(payload['messages'][-1]['content'])
        if name == 'plan_property_turn':
            p = plan(context['message'], 'KNOWLEDGE')
            if query:
                p['tasks'][0]['query'] = query
            if len(payload['messages']) == 3:
                catalog = json.loads(payload['messages'][1]['content'])['templates']
                hint = selection(catalog, target)
                if damage:
                    hint = damage(hint)
                p['template_selections'] = [{'task_index': 0, 'decision': hint}]
            return completion(p)
        assert name == 'match_property_templates', 'templates must bypass response generation'
        response = completion(selection(context['templates'], target))
        response['choices'][0]['message']['tool_calls'][0]['function']['name'] = name
        return response
    return send


@pytest.mark.parametrize('estate,q,target', [
    ('E001', '维保中心在哪儿呀？', BY_CASE['E001-GQ-010']['qa_id']),
    ('E001', '装修押金咋退呀？', BY_CASE['E001-GQ-040']['qa_id']),
    ('E001', '住宅物管费每平米多少？', BY_CASE['E001-GQ-019']['qa_id']),
    ('E002', '管家几点上班？', BY_CASE['E002-GQ-017']['qa_id']),
    ('E002', '晚上还能打电话联系管家吗？', BY_CASE['E002-GQ-018']['qa_id']),
])
def test_joint_call_reaches_literal_template_with_zero_generator(tmp_path, estate, q, target):
    calls, b = [], PacketBusiness(tmp_path)
    planner = LLMPlanner(LLMSettings(mode='online', api_key='test-only'), transport=model_transport(calls, target))
    session = AgentSession(b, b.token(estate), planner=planner)
    result = session.chat(q)
    row = next(r for r in ROWS if r['qa_id'] == target)
    assert result['text'] == row['answer']
    assert result['data']['debug_info']['semantic_request_strategy'] == 'JOINT_WITH_PLANNING'
    assert result['data']['generation_llm_call_count'] == 0 and len(calls) == 1
    assert not session._knowledge_matches and not b.orders
    catalog = json.loads(calls[0]['messages'][1]['content'])
    assert len(catalog['templates']) == 50 and catalog['estate_id'] == estate
    assert 'ADDITIONAL_QA' not in calls[0]['messages'][1]['content']
    session.close()


@pytest.mark.parametrize('enabled,expected', [(True, 1), (False, 2)])
def test_identical_answer_with_one_or_two_model_roundtrips(tmp_path, enabled, expected):
    calls, b = [], PacketBusiness(tmp_path)
    planner = LLMPlanner(LLMSettings(mode='online', api_key='test-only', joint_knowledge_planning=enabled),
        transport=model_transport(calls, BY_CASE['E001-GQ-010']['qa_id']))
    s = AgentSession(b, b.token(), planner=planner)
    result = s.chat('维保中心在哪儿呀？')
    assert result['text'] == BY_CASE['E001-GQ-010']['answer'] and len(calls) == expected
    assert result['data']['generation_llm_call_count'] == 0
    s.close()


@pytest.mark.parametrize('damage', [
    lambda d: {**d, 'matches': [{**d['matches'][0], 'version_hash': 'outdated'}]},
    lambda d: {**d, 'matches': [{**d['matches'][0], 'template_id': BY_CASE['E002-GQ-022']['qa_id']}]},
    lambda d: {**d, 'knowledge_ids': ['legacy-qa']},
    lambda d: {**d, 'unknown': 'untrusted'},
    lambda d: None,
])
def test_bad_optional_selection_uses_existing_matcher(tmp_path, damage):
    calls, b = [], PacketBusiness(tmp_path)
    p = LLMPlanner(LLMSettings(mode='online', api_key='test-only'),
        transport=model_transport(calls, BY_CASE['E001-GQ-010']['qa_id'], damage=damage))
    s = AgentSession(b, b.token(), planner=p)
    r = s.chat('维保中心在哪儿呀？')
    assert r['text'] == BY_CASE['E001-GQ-010']['answer'] and len(calls) == 2
    assert r['data']['debug_info']['semantic_request_strategy'] == 'SEPARATE'
    s.close()


def test_specific_home_conditions_still_require_separate_verified_match(tmp_path):
    b, calls = PacketBusiness(tmp_path), []
    p = LLMPlanner(LLMSettings(mode='online', api_key='test-only'),
        transport=model_transport(calls, BY_CASE['E002-GQ-006']['qa_id']))
    s = AgentSession(b, b.token('E002'), planner=p)
    r = s.chat('我家带电梯，物业费每平米多少钱？')
    assert len(calls) == 2 and r['data']['matched_template_id'] == BY_CASE['E002-GQ-006']['qa_id']
    context = json.loads(calls[1]['messages'][-1]['content'])['context']
    assert context['verified_conditions']['has_elevator'] is True
    assert '_joint_selections' not in context
    s.close()


def test_followup_and_correction_use_history_without_repeating_second_call(tmp_path):
    calls, b = [], PacketBusiness(tmp_path)
    p = LLMPlanner(LLMSettings(mode='online', api_key='test-only'), transport=model_transport(calls, BY_CASE['E002-GQ-018']['qa_id']))
    s = AgentSession(b, b.token('E002'), planner=p)
    previous = next(r for r in ROWS if r['qa_id'] == BY_CASE['E002-GQ-017']['qa_id'])
    s.chat(previous['question'])
    for q in ('那电话呢？', '你弄错了，我说的是电话联系时间'):
        r = s.chat(q)
        assert r['data']['matched_template_id'] == BY_CASE['E002-GQ-018']['qa_id']
        assert r['data']['debug_info']['semantic_request_strategy'] == 'JOINT_WITH_PLANNING'
    assert len(calls) == 2 and len(s.conversation_state['conversation_history']) == 6
    assert calls[0]['messages'][:2] == calls[1]['messages'][:2]  # stable provider-cache prefix
    assert json.loads(calls[1]['messages'][-1]['content'])['conversation_history'][-1]['role'] == 'assistant'
    s.close()


def test_live_house_access_is_checked_even_when_joint_model_matches(tmp_path, monkeypatch):
    b, calls = PacketBusiness(tmp_path), []
    p = LLMPlanner(LLMSettings(mode='online', api_key='test-only'), transport=model_transport(calls, BY_CASE['E001-GQ-010']['qa_id']))
    def denied(*args, **kwargs):
        raise POCError('RESOURCE_NOT_AVAILABLE', '房屋关联失效', 404)
    monkeypatch.setattr(b._repo, 'eligible_unit', denied)
    p.respond = lambda *a, **kw: text_reply('这项信息暂时无法核实。')
    s = AgentSession(b, b.token(), planner=p)
    r = s.chat('维保中心在哪儿呀？')
    assert r['data'].get('matched_template_id') is None
    assert 'RESOURCE_NOT_AVAILABLE' in r['data']['warnings']
    assert not b.orders
    s.close()


def test_planner_rewrite_cannot_turn_narrower_question_into_exact_approved_match(tmp_path):
    b, calls = PacketBusiness(tmp_path), []
    q = '晚上十点管家能上门吗？'
    canonical = next(r for r in ROWS if r['qa_id'] == BY_CASE['E002-GQ-017']['qa_id'])['question']
    p = LLMPlanner(LLMSettings(mode='online', api_key='test-only'),
        transport=model_transport(calls, BY_CASE['E002-GQ-017']['qa_id'], query=canonical,
            damage=lambda d: {'matches': [{**d['matches'][0], 'coverage': 'PARTIAL',
                'question_relation': 'NARROWER', 'additional_fact_required': True}], 'clarification': 'CONTACT_MODE'}))
    p.respond = lambda *a, **kw: text_reply('晚上十点能否上门还没有核实。')
    s = AgentSession(b, b.token('E002'), planner=p)
    r = s.chat(q)
    assert r['data']['status'] == 'NEED_CLARIFICATION' and not r['data']['matched_template_id']
    assert len(calls) == 1
    s.close()


def test_template_interlude_retains_draft_and_cancel_still_revokes_card(tmp_path):
    b, calls = PacketBusiness(tmp_path), []
    s = AgentSession(b, b.token())
    preview = s.chat('帮我报修，模拟厨房水龙头滴水')
    pending = copy.deepcopy(s.pending)
    p = LLMPlanner(LLMSettings(mode='online', api_key='test-only'), transport=model_transport(calls, BY_CASE['E001-GQ-010']['qa_id']))
    s.planner = p
    assert s.chat('维保中心在哪儿呀？')['text'] == BY_CASE['E001-GQ-010']['answer']
    assert s.pending == pending and len(calls) == 1
    s.chat('先别提交')
    assert s.pending is None and s.confirm(preview['ticket'], accept=True)['kind'] == 'ERROR'
    assert not b.orders and not s._knowledge_matches
    s.close()


def test_joint_ui_refresh_neither_repeats_answer_nor_requests_model(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from property_ontology.agent import ui_backend
    from property_ontology.config import ROOT
    b, calls = PacketBusiness(tmp_path), []
    p = LLMPlanner(LLMSettings(mode='online', api_key='test-only'), transport=model_transport(calls, BY_CASE['E001-GQ-010']['qa_id']))
    monkeypatch.setattr(ui_backend, 'open_session', lambda e,u,pw: AgentSession(b,b.token(e,u),planner=p))
    app = AppTest.from_file(str(ROOT/'src/property_ontology/streamlit_app.py'), default_timeout=20).run()
    app.text_input(key='login_password').input('mock-password')
    app.button[0].click().run()
    app.chat_input[0].set_value('维保中心在哪儿呀？').run()
    for _ in range(3):
        app.run()
        assert not app.exception and len(app.chat_message) == 2 and len(calls) == 1
        assert [t.value for t in app.text].count(BY_CASE['E001-GQ-010']['answer']) == 1
    assert not b.orders


@pytest.mark.parametrize('changed', ['estate', 'query', 'history', 'subject', 'version', 'answer', 'conditions'])
def test_one_turn_match_is_invalidated_by_any_relevant_context_change(changed):
    from property_ontology.agent.knowledge_planning import BoundSelection, frozen, signature
    from property_ontology.config import ROOT
    from property_ontology.qa.day1_poc import load, projected
    rows, receipt, _ = load(ROOT)
    catalog = [projected(r, receipt) for r in rows['E001']]
    ctx = {'knowledge_set': 'ONTOLOGY_PACKAGE_POC', 'estate_id': 'E001', 'resolved_query': 'where',
           'conversation_history': [], 'knowledge_context': None}
    bound = BoundSelection('E001', 'where?', 'where', signature(catalog), frozen([]), frozen(None), '{}')
    assert bound.reuse('where?', ctx, catalog) is not None
    if changed == 'estate':
        ctx['estate_id'] = 'E002'
    elif changed == 'query':
        ctx['resolved_query'] = 'when'
    elif changed == 'history':
        ctx['conversation_history'] = [{'role': 'user', 'content': 'new context'}]
    elif changed == 'subject':
        ctx['knowledge_context'] = {'question': 'a different service'}
    elif changed == 'version':
        catalog[0]['version_hash'] = 'new-version'
    elif changed == 'answer':
        catalog[0]['standard_answer'] = 'changed source'
    else:
        ctx['verified_conditions'] = {'has_elevator': True}
    assert bound.reuse('where?', ctx, catalog) is None


def test_measuring_budget_counts_failed_attempts_and_survives_restart(tmp_path):
    import importlib.util

    from property_ontology.config import ROOT
    spec = importlib.util.spec_from_file_location('measure_latency', ROOT/'scripts/check_day1_latency.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    count = []
    class Down:
        def _http(self, payload):
            count.append(1)
            raise POCError('LLM_UNAVAILABLE', 'stub', 503)
    payload = {'tools': [{'function': {'name': 'plan_property_turn'}}]}
    path = tmp_path/'attempts.json'
    with pytest.raises(POCError, match='stub'):
        module.Budget(path, 1).call(Down(), payload)
    with pytest.raises(POCError, match='调用上限'):
        module.Budget(path, 1).call(Down(), payload)
    assert len(count) == 1 and json.loads(path.read_text())[0]['status'] == 'FAILED'
