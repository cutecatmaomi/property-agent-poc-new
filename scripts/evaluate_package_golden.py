"""Actual Agent/Actions/Neo4j chain; bounded paid calls, no business submissions.

Gold questions are evaluation-only input, never matching rules or runtime aliases.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from threading import Lock
import time

from property_ontology.actions import PropertyActions
from property_ontology.agent import AgentSession
from property_ontology.agent.llm import LLMPlanner, LLMSettings
from property_ontology.config import ROOT
from property_ontology.domain import POCError
from property_ontology.runtime import RuntimeSettings

OUTPUT = ROOT/'outputs/golden_accuracy_fix_20261103'


class Budget:
    def __init__(self, maximum):
        self.maximum, self.lock = maximum, Lock()
        self.path = OUTPUT/'model_request_ledger.jsonl'
        self.used = len(self.path.read_text().splitlines()) if self.path.exists() else 0

    def take(self, case_id, payload):
        with self.lock:
            if self.used >= self.maximum:
                raise POCError('LLM_BUDGET_EXHAUSTED', '本轮请求额度已用完。', 503)
            self.used += 1
            record = {'request_number': self.used, 'case_id': case_id,
                      'function': payload.get('tool_choice', {}).get('function', {}).get('name'),
                      'time': time.time()}
            with self.path.open('a') as f:
                f.write(json.dumps(record, ensure_ascii=False)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--online', action='store_true', help='Explicit paid-model opt-in required')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument('--max-requests', type=int, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--name', default='live_golden')
    parser.add_argument('--canonical', action='store_true')
    args = parser.parse_args()
    if not args.online or args.max_requests <= 0:
        parser.error('需要显式 --online 和正数 --max-requests；自然问法回归会调用付费模型。')
    global OUTPUT
    OUTPUT = args.output_dir.resolve()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT/(args.name+'.jsonl')
    if result_path.exists():
        raise SystemExit('Existing result preserved; choose a new --name.')
    import sys
    sys.path.insert(0, str(ROOT))
    from tools.handoff.golden_questions import bound_cases
    rows = [{**r, 'policy_id': r['expected_policy_id'], 'question': r['simulated_question']} for r in bound_cases(ROOT)]
    if args.canonical:
        from property_ontology.qa.day1_poc import load
        packet = load(ROOT)
        catalog = {r.get('policy_id',r['qa_id']): r for rs in packet[0].values() for r in rs}
        rows = [{**r, 'question': catalog[r['policy_id']]['question']} for r in rows]
    rows = rows[args.start:args.end]
    budget = Budget(args.max_requests)
    settings = RuntimeSettings.load()
    credentials = json.loads((settings.auth_path.parent/'mock_credentials.json').read_text())
    model_settings = LLMSettings.load()
    observer = PropertyActions(settings)
    before = {e: observer._repo.stats(e) for e in ('E001', 'E002')}

    def run(r):
        app = PropertyActions(settings)
        p = LLMPlanner(model_settings)
        calls = []
        def transport(payload):
            budget.take(r['case_id'], payload)
            calls.append(payload.get('tool_choice', {}).get('function', {}).get('name'))
            return p._http(payload)
        p._transport = p._response_transport = transport
        s = None
        try:
            c = next(c for c in credentials if c['estate_id'] == r['estate_id'] and c['username'] == 'owner1')
            token = app.login(r['estate_id'], 'owner1', c['password'])['token']
            s = AgentSession(app, token, planner=p)
            started = time.monotonic()
            reply = s.chat(r['question'])
            elapsed = round(time.monotonic()-started, 3)
            d = reply.get('data', {})
            actual = d.get('matched_policy_ids', [])
            expected = r.get('expected_policy_ids', [r['policy_id']] if r.get('policy_id') else [])
            passed = (r['policy_id'] in actual and r['policy_statement'] in reply['text']
                      and d.get('generation_llm_called') is False
                      and reply['text'] == r['answer']) if r.get('policy_id') else not actual and reply['kind'] != 'ERROR'
            return {'case_id':r['case_id'], 'estate_id':r['estate_id'], 'question':r['question'],
                    'expected_policy_ids':expected, 'matched_policy_ids':actual,
                    'final_answer':reply['text'], 'answer_source':d.get('answer_source'),
                    'reference_answer':r.get('reference_answer'), 'reference_answer_exact':reply['text']==r.get('reference_answer'),
                    'generation_llm_called':d.get('generation_llm_called',False), 'requests':calls,
                    'seconds':elapsed, 'passed':passed, 'reply':reply}
        except Exception as exc:
            return {'case_id':r['case_id'], 'question':r['question'], 'passed':False,
                    'error':getattr(exc,'code',type(exc).__name__), 'requests':calls}
        finally:
            if s: s.close()
            p.close()
            app.close()

    results = []
    with result_path.open('x') as log, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run, r) for r in rows]):
            r = future.result(); results.append(r)
            log.write(json.dumps(r, ensure_ascii=False)+'\n'); log.flush()
            print(json.dumps({k:r.get(k) for k in ('case_id','passed','seconds','error')},ensure_ascii=False),flush=True)
    after = {e: observer._repo.stats(e) for e in ('E001', 'E002')}
    observer.close()
    summary = {'question_bank':'GoldenQuestions.xlsx', 'canonical_mode':args.canonical, 'total':len(results), 'passed':sum(r['passed'] for r in results),
               'accuracy':sum(r['passed'] for r in results)/max(1,len(results)),
               'paid_requests_so_far':budget.used, 'graph_counts_unchanged':before==after,
               'graph_before':before,'graph_after':after,
               'method':'Actual AgentSession.chat → LangGraph → authenticated Actions → selector → presenter. Gold policy ID+fact required; generator must not rewrite fact.'}
    result_path.with_suffix('.summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
