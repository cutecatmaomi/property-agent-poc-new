"""Actual Golden Questions loader bound to unchanged, existing POC reviews."""
import copy
import hashlib
import json
from pathlib import Path

SET = 'GOLDEN_QUESTIONS_100'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def require(value):
    if not value:
        raise ValueError('Golden Questions mapping or authorization invalid')


def load(root, config):
    from .package_poc import load as load_reviewed_graph
    root = Path(root)
    require(config['active_set'] == SET and config['legacy_fallback'] is False)
    upstream = config['upstream_config']
    require(upstream['active_set'] == 'ONTOLOGY_PACKAGE_POC' and upstream['legacy_fallback'] is False)
    spec = config['golden_questions']
    path = (root/spec['path']).resolve()
    require(path.is_relative_to(root.resolve()))
    content = path.read_bytes()
    require(digest(content) == spec['sha256'])
    epoch = {'question_bank_sha256': spec['sha256'], 'upstream_manifest_sha256': upstream['manifest_sha256']}
    require(digest(json.dumps(epoch, sort_keys=True, separators=(',', ':')).encode()) == config['manifest_sha256'])
    data = json.loads(content)
    require(data['question_set_id'] == SET and data['question_source'] == 'GoldenQuestions.xlsx')
    items = data['questions']
    require(len(items) == 100 and len({q['question_id'] for q in items}) == 100)
    # Verify actual source bytes, graph links, review purpose and exact versions.
    source_rows, source_receipt, _ = load_reviewed_graph(root, upstream)
    by_policy = {r['qa_id']: r for records in source_rows.values() for r in records}
    require({q['linked_policy_ids'][0] for q in items} == set(by_policy))
    rows, proofs, refs = {'E001': [], 'E002': []}, {}, {}
    for q in items:
        estate = q['estate_id']; identity = q['question_id']
        require(estate in rows and q['template_id'] == identity and identity.startswith(estate+'-GQ-'))
        require(q['question_set_id'] == SET and q['source_type'] == 'GOLDEN_QUESTIONS')
        require(len(q['linked_policy_ids']) == 1 and q['canonical_question'].strip())
        policy_id = q['linked_policy_ids'][0]
        original = by_policy[policy_id]
        proof = source_receipt['records'][policy_id]
        require(original['estate_id'] == estate and q['policy_version_hash'] == proof['record_sha256'])
        require(q['authorized_answer'] == original['answer'] and q['conditions'] == original['conditions'])
        require(set(q['source_evidence_ids']) == {r['chunk_id'] for r in proof['evidence_refs']})
        require(set(q['source_document_ids']) == {r['document_id'] for r in proof['evidence_refs']})
        require(set(q['source_review_ids']) <= {r['review_decision_id'] for r in proof['review_events']})
        if q['standard_answer'] != original['answer']:
            require(identity == 'E001-GQ-009' and q['requires_operation_state_check'] is True)
            require(q['standard_answer'] == original['answer'] + '\n这是包内资料的处理流程描述，本次未联系维修人员或提交工单。')
        row = {**original, 'qa_id': identity, 'policy_id': policy_id, 'question_id': identity,
               'question': q['canonical_question'], 'answer': q['authorized_answer'],
               'reference_answer': q['standard_answer'], 'source_type': 'GOLDEN_QUESTIONS',
               'origin': 'GOLDEN_QUESTION_BOUND_TO_EXISTING_POC_POLICY', 'question_set_id': SET,
               'workbook_row': q['workbook_provenance']['row']}
        rows[estate].append(row)
        proofs[identity] = {**copy.deepcopy(proof), 'workbook_row': q['workbook_provenance']['row']}
        refs[identity] = copy.deepcopy(source_receipt['graph_references'][policy_id])
    require({e:len(v) for e,v in rows.items()} == {'E001':50,'E002':50})
    receipt = {**source_receipt, 'knowledge_set': SET, 'packet_sha256': config['manifest_sha256'],
               'records': proofs, 'graph_references': refs, 'upstream_knowledge_set': upstream['active_set'],
               'golden_questions_sha256': spec['sha256']}
    return rows, receipt, config
