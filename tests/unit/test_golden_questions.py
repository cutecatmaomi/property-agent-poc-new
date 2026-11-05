"""Actual conversation graph + controlled Action + approved packet, no legacy retrieval."""
import copy
from datetime import timedelta
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
from golden_support import ROWS, BY_CASE, PacketBusiness, NoModel
from test_steward_conversation import ScriptedPlanner, plan, text_reply, factual_reply

from property_ontology.actions import PropertyActions
from property_ontology.agent import AgentSession
from property_ontology.config import ROOT
from property_ontology.domain import POCError
from property_ontology.qa.day1_poc import load, projected
from property_ontology.qa.day1_poc import FOLDER, SET
from property_ontology.qa.templates import SemanticDecision



@pytest.mark.parametrize('row', ROWS, ids=[r['qa_id'] for r in ROWS])
def test_all_100_through_graph_action_selector_and_presenter(tmp_path, row):
    b = PacketBusiness(tmp_path)
    s = AgentSession(b, b.token(row['estate_id']), planner=NoModel())
    reply = s.chat(row['question'])
    assert reply['text'] == row['answer'], reply
    assert reply['data']['answer_text'] == row['answer']
    assert reply['data']['matched_template_id'] == row['qa_id']
    assert reply['data']['answer_source'] == 'TEMPLATE'
    assert reply['data']['generation_llm_call_count'] == 0
    assert reply['data']['generation_llm_called'] is False
    assert not b.orders
    assert reply['data']['debug_info']['legacy_retrieval_called'] is False
    assert reply['data']['evidence_refs'][0]['document_id'].startswith(row['estate_id'] + ':')
    if reply['data'].get('quote_notice'):
        assert '本次没有' in s.conversation_state['conversation_history'][-1]['content']
    s.close()


def semantic_planner(turns, target):
    p = ScriptedPlanner(turns)
    def match(q, *, context, templates, knowledge):
        assert not knowledge and len(templates) == 50
        t = next(t for t in templates if t['template_id'] == target)
        return SemanticDecision(matches=[{'template_id': target, 'version_hash': t['version_hash'],
            'coverage': 'FULL', 'score': .99, 'reason': 'test semantic model', 'question_relation': 'EQUIVALENT', 'additional_fact_required': False}])
    p.match_templates = match
    return p


def test_semantic_followup_keeps_estate_and_context(tmp_path):
    b = PacketBusiness(tmp_path)
    q = '那电话呢'
    p = semantic_planner([(plan(q, 'KNOWLEDGE'), None)], BY_CASE['E002-GQ-018']['qa_id'])
    s = AgentSession(b, b.token('E002'), planner=p)
    row = next(r for r in ROWS if r['qa_id'] == BY_CASE['E002-GQ-017']['qa_id'])
    s.chat(row['question'])
    r = s.chat(q)
    assert p.contexts[-1]['conversation_history'][-1]['content'] == row['answer']
    assert r['data']['matched_template_id'] == BY_CASE['E002-GQ-018']['qa_id'] and not r['data']['generation_llm_called']


def test_no_template_natural_reply_and_no_legacy_fallback(tmp_path):
    b = PacketBusiness(tmp_path)
    q = '合同什么时候可以拿'
    p = ScriptedPlanner([(plan(q, 'KNOWLEDGE'), text_reply('领取合同的时间还没有核实，我暂时不能确定。'))])
    p.match_templates = lambda *a, **kw: SemanticDecision()
    s = AgentSession(b, b.token(), planner=p)
    r = s.chat(q)
    assert r['data']['answer_source'] == 'GENERAL_CONVERSATION'
    assert r['data']['generation_llm_called'] and '没有核实' in r['text']
    assert not r['data']['evidence_refs']


def test_current_mode_wrong_estate_wrong_unit_and_date_are_denied(tmp_path):
    b = PacketBusiness(tmp_path)
    token = b.token()
    q = ROWS[0]['question']
    assert b.answer_question(token, q, knowledge_mode='current')['status'] == 'NOT_VERIFIED'
    assert b.answer_question(token, '国际健康生态谷' + q, knowledge_mode='demo')['status'] == 'NEED_CLARIFICATION'
    with pytest.raises(POCError):
        b.answer_question(token, q, knowledge_mode='demo', unit_id='E002:UNIT:MOCK-101')
    b.now = b.now.replace(year=2026, month=10, day=27)  # before package POC authorization
    token = b.token()
    assert b.answer_question(token, q, knowledge_mode='demo')['status'] == 'NOT_VERIFIED'


@pytest.mark.parametrize('target', ['questions', 'config', 'upstream_graph', 'source'])
def test_tampering_fails_closed_without_legacy_read(tmp_path, target):
    root = tmp_path/'root'; (root/'config').mkdir(parents=True)
    shutil.copyfile(ROOT/'config/knowledge_source.json', root/'config/knowledge_source.json')
    shutil.copytree(ROOT/FOLDER, root/FOLDER)
    from property_ontology.qa.package_poc import FOLDER as UPSTREAM_FOLDER
    shutil.copytree(ROOT/UPSTREAM_FOLDER, root/UPSTREAM_FOLDER)
    packet = load(ROOT)
    for src in packet[1]['sources_by_path'].values():
        file = root/src['path']; file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/src['path'], file)
    relative = (next(iter(packet[1]['sources_by_path'])) if target == 'source' else
                'config/knowledge_source.json' if target == 'config' else
                UPSTREAM_FOLDER/'E001.graph.json' if target == 'upstream_graph' else
                FOLDER/'GoldenQuestions_100问答.json')
    file = root/relative
    if target == 'config':
        changed = json.loads(file.read_text()); changed['active_set'] = 'DISABLED_OLD_BANK'
        file.write_text(json.dumps(changed))
    else:
        file.write_bytes(file.read_bytes()+b' ')
    business = PacketBusiness(tmp_path, root)
    with pytest.raises(POCError):
        business.answer_question(business.token(), ROWS[0]['question'], knowledge_mode='demo')


def test_template_interlude_does_not_reset_draft_confirm_or_order(tmp_path):
    b = PacketBusiness(tmp_path)
    msg = '帮我报修，厨房漏水'
    p = ScriptedPlanner([(plan(msg, 'REPAIR_PREPARE', action_request='EXPLICIT', repair={'location':'厨房','symptom':'漏水'}), text_reply('核对卡片后再提交。')),
                         (plan('查询我的工单进度', 'TRACK_ORDER'), factual_reply(''))])
    s = AgentSession(b, b.token(), planner=p)
    preview = s.chat(msg); pending = copy.deepcopy(s.pending)
    s.chat(ROWS[0]['question'])
    assert s.pending == pending and not b.orders
    created = s.confirm(preview['ticket'], accept=True)
    assert created['kind'] == 'ORDER_CREATED' and len(b.orders) == 1
    assert s.confirm(preview['ticket'], accept=True)['data']['replayed']
    assert created['data']['number'] in s.chat('查询我的工单进度')['text']


def test_offline_canonical_interlude_preserves_pending_draft(tmp_path):
    b = PacketBusiness(tmp_path)
    s = AgentSession(b, b.token())
    preview = s.chat('我要报修，模拟厨房水龙头滴水')
    pending = copy.deepcopy(s.pending)
    assert preview['kind'] == 'REPAIR_PREVIEW'
    assert s.chat(ROWS[0]['question'])['text'] == ROWS[0]['answer']
    assert s.pending == pending and not b.orders
    assert s.confirm(preview['ticket'], accept=True)['kind'] == 'ORDER_CREATED'


def test_match_uses_original_scope_and_keeps_resolved_query_as_context(tmp_path):
    b = PacketBusiness(tmp_path)
    original, expanded = '装修交的押金怎么才能退？', '装修押金退还的条件和流程'
    decision = plan(original, 'KNOWLEDGE')
    decision['tasks'][0]['query'] = expanded
    p = semantic_planner([(decision, None)], BY_CASE['E001-GQ-040']['qa_id'])
    previous = p.match_templates
    def match(q, **kwargs):
        assert q == original
        assert kwargs['context']['resolved_query'] == expanded
        return previous(q, **kwargs)
    p.match_templates = match
    reply = AgentSession(b, b.token(), planner=p).chat(original)
    assert reply['data']['matched_template_id'] == BY_CASE['E001-GQ-040']['qa_id']
    assert not reply['data']['generation_llm_called']


def test_knowledge_correction_preserves_unsubmitted_repair(tmp_path):
    b = PacketBusiness(tmp_path)
    first = '帮我报修，厨房漏水'
    correction = '你弄错了，我问的是电话联系，不是现场上班'
    p = semantic_planner([
        (plan(first, 'REPAIR_PREPARE', action_request='EXPLICIT', repair={'location':'厨房','symptom':'漏水'}), text_reply('核对卡片后再提交。')),
        ({'route':'CORRECTION', 'context_operation':'CORRECT'}, None),
    ], BY_CASE['E002-GQ-018']['qa_id'])
    s = AgentSession(b, b.token('E002'), planner=p)
    preview = s.chat(first)
    row = next(r for r in ROWS if r['qa_id'] == BY_CASE['E002-GQ-017']['qa_id'])
    s.chat(row['question'])
    pending = copy.deepcopy(s.pending)
    reply = s.chat(correction)
    assert reply['data']['matched_template_id'] == BY_CASE['E002-GQ-018']['qa_id']
    assert s.pending == pending and not b.orders
    assert s.confirm(preview['ticket'], accept=True)['kind'] == 'ORDER_CREATED'


def test_missing_config_with_packet_cannot_reenable_legacy(tmp_path):
    root = tmp_path/'root'
    shutil.copytree(ROOT/FOLDER, root/FOLDER)
    b = PacketBusiness(tmp_path, root)
    with pytest.raises(POCError, match='配置缺失'):
        b.answer_question(b.token(), ROWS[0]['question'], knowledge_mode='demo')


def test_online_acceptance_budget_counts_protocol_retries():
    import importlib.util
    spec = importlib.util.spec_from_file_location('day1_acceptance', ROOT/'scripts/check_day1_100.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    planner = object.__new__(module.CountingPlanner)
    planner.maximum = 1
    planner._count('semantic')
    with pytest.raises(POCError, match='预算已用完'):
        planner._count('semantic')
    assert sum(module.CountingPlanner.calls.values()) == 1


@pytest.mark.parametrize('failure', ['CONFLICT', 'VERSION', 'FOREIGN', 'PARTIAL'])
def test_semantic_rejection_never_returns_unverified_template(tmp_path, failure):
    b = PacketBusiness(tmp_path)
    class Matcher:
        def match_templates(self, q, *, templates, **kwargs):
            t = templates[0]
            if failure == 'CONFLICT':
                return SemanticDecision(conflict_template_ids=[t['template_id']])
            return SemanticDecision(matches=[{'template_id': BY_CASE['E002-GQ-022']['qa_id'] if failure == 'FOREIGN' else t['template_id'],
                'version_hash': 'outdated' if failure == 'VERSION' else t['version_hash'],
                'coverage': 'PARTIAL' if failure == 'PARTIAL' else 'FULL', 'score': .99,
                'reason': 'invalid match test', 'question_relation': 'EQUIVALENT', 'additional_fact_required': failure == 'PARTIAL'}])
    r = b.answer_question(b.token(), '中心在哪儿呢', knowledge_mode='demo', semantic_matcher=Matcher())
    assert r['status'] != 'ANSWERED' and r['matched_template_id'] is None


def test_new_ui_rerun_neither_repeats_answer_nor_calls_model(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from property_ontology.agent import ui_backend
    b = PacketBusiness(tmp_path)
    monkeypatch.setattr(ui_backend, 'open_session', lambda e,u,p: AgentSession(b,b.token(e,u),planner=NoModel()))
    app = AppTest.from_file(str(ROOT/'src/property_ontology/streamlit_app.py'), default_timeout=20).run()
    app.selectbox(key='login_estate').select('E002')
    app.text_input(key='login_password').input('mock-password')
    app.button[0].click().run()
    row = next(r for r in ROWS if r['qa_id'] == BY_CASE['E002-GQ-041']['qa_id'])
    app.chat_input[0].set_value(row['question']).run()
    for _ in range(3):
        app.run()
        assert not app.exception and len(app.chat_message) == 2
        assert [t.value for t in app.text].count(row['answer']) == 1
        assert not b.orders  # standard fact is not an executed service promise
    assert not b.orders
