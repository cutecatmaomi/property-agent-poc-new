"""Actual Golden loader, planning catalog and response context consistency."""
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
from golden_support import PacketBusiness, NoModel
from property_ontology.agent import AgentSession
from property_ontology.agent.dialogue import decision_context
from property_ontology.agent.reply_generation import response_context
from property_ontology.agent.knowledge_planning import BoundSelection, signature, frozen
from property_ontology.qa.day1_poc import load, projected, configuration
from property_ontology.config import ROOT
from property_ontology.domain import POCError


def test_config_and_loaded_question_catalog_are_golden_not_policy_titles():
    rows,receipt,config=load(ROOT)
    bank=json.loads((ROOT/config['golden_questions']['path']).read_text())
    assert config['active_set']=='GOLDEN_QUESTIONS_100' and config['legacy_fallback'] is False
    by_id={r['qa_id']:r for rs in rows.values() for r in rs}
    assert set(by_id)=={q['question_id'] for q in bank['questions']}
    for q in bank['questions']:
        r=by_id[q['question_id']]
        assert r['question']==q['canonical_question']
        assert r['answer']==q['authorized_answer']
        assert r['policy_id']==q['linked_policy_ids'][0]
        assert projected(r,receipt)['source_type']=='GOLDEN_QUESTIONS'


def test_planning_reply_and_exact_answer_share_same_question_set(tmp_path):
    business=PacketBusiness(tmp_path)
    session=AgentSession(business,business.token('E002'),planner=NoModel())
    try:
        question='楼栋管家每天在现场提供服务的时间是几点到几点？'
        reply=session.chat(question)
        d=reply['data']
        assert d['matched_template_id']=='E002-GQ-017'
        assert d['source_type']=='GOLDEN_QUESTIONS'
        assert d['knowledge_set']=='GOLDEN_QUESTIONS_100'
        assert d['debug_info']['golden_question_id']=='E002-GQ-017'
        assert d['generation_llm_call_count']==0
        for context in [decision_context(session),response_context(session,reply)]:
            assert context['knowledge_set']==context['question_set_id']=='GOLDEN_QUESTIONS_100'
    finally:session.close()


def test_old_question_sets_are_rejected_in_configuration(tmp_path):
    (tmp_path/'config').mkdir()
    for old in ['DAY1_100_POC','ONTOLOGY_PACKAGE_POC']:
        (tmp_path/'config/knowledge_source.json').write_text(json.dumps({'active_set':old,'legacy_fallback':False}))
        with pytest.raises(POCError):configuration(tmp_path)


def test_existing_golden_bound_selection_can_be_reused_without_extra_matching():
    rows,receipt,_=load(ROOT)
    catalog=[projected(r,receipt) for r in rows['E001']]
    ctx={'knowledge_set':'GOLDEN_QUESTIONS_100','estate_id':'E001','resolved_query':'where',
         'conversation_history':[],'knowledge_context':None}
    bound=BoundSelection('E001','where?','where',signature(catalog),frozen([]),frozen(None),'{}')
    assert bound.reuse('where?',ctx,catalog) is not None
    ctx['knowledge_set']='DAY1_100_POC'
    assert bound.reuse('where?',ctx,catalog) is None
