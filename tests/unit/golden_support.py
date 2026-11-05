"""Current GoldenQuestions workbook + isolated business doubles, no archived bank."""
from datetime import timedelta
from types import SimpleNamespace
import pytest
from test_day3_agent import FakeBusiness
from property_ontology.actions import PropertyActions
from property_ontology.config import ROOT
from property_ontology.domain import POCError
from tools.handoff.golden_questions import bound_cases

ROWS = bound_cases(ROOT)
BY_CASE = {r['case_id']: r for r in ROWS}

class ExclusiveRepo:
    def resident_active(self, *args):
        return True

    def eligible_unit(self, who, unit_id, day, **kwargs):
        if not unit_id.startswith(who.estate_id + ':'):
            raise POCError('RESOURCE_NOT_AVAILABLE', '房屋不可用', 404)
        return {'unit': {'has_elevator': True, 'residence_category': 'RESIDENTIAL'}, 'building': {'building_form': 'HIGH_RISE'}}

    def __getattr__(self, key):
        raise AssertionError('legacy graph knowledge must not be read: ' + key)


class PacketBusiness(FakeBusiness):
    _run = PropertyActions._run
    query_knowledge = PropertyActions.query_knowledge
    answer_question = PropertyActions.answer_question

    def __init__(self, tmp_path, root=ROOT):
        super().__init__(tmp_path)
        self.now += timedelta(days=1)
        self.settings = SimpleNamespace(root=root)
        self._repo = ExclusiveRepo()


class NoModel:
    enable_natural_reply = True

    def decide(self, *args, **kwargs):
        pytest.fail('canonical approved question should bypass planning model')

    def respond(self, *args, **kwargs):
        pytest.fail('template final answer must not use a generator')


