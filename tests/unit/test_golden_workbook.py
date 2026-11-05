"""Excel is the only current test bank, not an authority to approve runtime facts."""
import json
from pathlib import Path
import shutil
import zipfile
import xml.etree.ElementTree as ET
import pytest

from golden_support import ROWS, PacketBusiness
from test_golden_questions import semantic_planner
from test_steward_conversation import plan
from property_ontology.agent import AgentSession
from property_ontology.config import ROOT
from property_ontology.qa.day1_poc import load
from tools.handoff.golden_questions import WORKBOOK, NS, bound_cases, read_cases


@pytest.mark.parametrize('row', ROWS, ids=[r['case_id'] for r in ROWS])
def test_spreadsheet_wording_through_agent_with_scripted_semantic_selection(tmp_path, row):
    # Explicit test double, not a live semantic-accuracy claim: upstream chooses
    # the expected policy, downstream must preserve scope, evidence and text.
    business = PacketBusiness(tmp_path)
    planner = semantic_planner([(plan(row['simulated_question'], 'KNOWLEDGE'), None)], row['qa_id'])
    session = AgentSession(business, business.token(row['estate_id']), planner=planner)
    try:
        reply = session.chat(row['simulated_question'])
        assert reply['data']['matched_template_id'] == row['case_id']
        assert reply['data']['matched_policy_ids'] == [row['expected_policy_id']]
        assert reply['text'] == row['answer']
        assert reply['data']['generation_llm_called'] is False
        assert reply['data']['generation_llm_call_count'] == 0
        assert reply['data']['debug_info']['legacy_retrieval_called'] is False
        assert not business.orders
    finally:
        session.close()


def test_workbook_counts_scope_versions_and_reference_difference_are_explicit():
    assert len(ROWS) == 100
    assert {e:sum(r['estate_id']==e for r in ROWS) for e in ['E001','E002']} == {'E001':50,'E002':50}
    assert all(r['source_row'] and r['evidence_row'] and r['authorized_source_path'] for r in ROWS)
    differing = [r for r in ROWS if not r['reference_answer_matches_authorized']]
    assert [r['case_id'] for r in differing] == ['E001-GQ-009']
    assert differing[0]['reference_answer'].startswith(differing[0]['answer'])


@pytest.mark.parametrize('sheet,cell,value,expected', [
    ('sheet1.xml','D8','', 'Missing question'),
    ('sheet1.xml','B8','E002 青岛国际健康生态谷', 'Estate/reference'),
    ('sheet1.xml','G8','DOCX 正文段落 999', 'Main/reference source'),
    ('sheet2.xml','L8','wrong-version', 'mapping invalid'),
    ('sheet2.xml','H8','E001:CHUNK:missing', 'mapping invalid'),
    ('sheet2.xml','G8','E001:POL:missing', 'policy coverage'),
])
def test_modified_workbook_cannot_replace_missing_or_unapproved_facts(tmp_path, sheet, cell, value, expected):
    book = tmp_path/'GoldenQuestions.xlsx'
    with zipfile.ZipFile(ROOT/WORKBOOK) as src, zipfile.ZipFile(book, 'w') as dst:
        for entry in src.infolist():
            data = src.read(entry.filename)
            if entry.filename == 'xl/worksheets/'+sheet:
                xml = ET.fromstring(data)
                item = next(c for c in xml.findall('.//s:c', NS) if c.attrib.get('r') == cell)
                for child in list(item): item.remove(child)
                item.set('t','inlineStr')
                ET.SubElement(ET.SubElement(item, '{'+NS['s']+'}is'), '{'+NS['s']+'}t').text = value
                data = ET.tostring(xml)
            dst.writestr(entry, data)
    with pytest.raises(ValueError, match=expected):
        bound_cases(ROOT, book)
    # Test-bank edits never enter the runtime loader or change approvals.
    rows, _, config = load(ROOT)
    assert config['active_set'] == 'GOLDEN_QUESTIONS_100'
    assert sum(map(len, rows.values())) == 100
