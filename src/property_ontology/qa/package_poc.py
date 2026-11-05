"""Read-only POC graph snapshot, with existing human reviews and live source hashes.

The evaluation spreadsheet and its expected answers are deliberately never read here.
This adapter exposes package facts through the existing semantic/template pipeline.
"""
import hashlib
import json
from pathlib import Path

SET = 'ONTOLOGY_PACKAGE_POC'
FOLDER = Path('data/knowledge_sets') / SET


def sha(data):
    return hashlib.sha256(data).hexdigest()


def check(value):
    if not value:
        raise ValueError('package proof invalid')


def load(root, config):
    root = Path(root)
    content = (root/FOLDER/'manifest.json').read_bytes()
    check(sha(content) == config['manifest_sha256'])
    manifest = json.loads(content)
    rows, proofs, references, sources = {}, {}, {}, {}
    for estate in ('E001', 'E002'):
        spec = manifest['graphs'][estate]
        graph_bytes = (root/spec['path']).read_bytes()
        check(sha(graph_bytes) == spec['sha256'])
        graph = json.loads(graph_bytes)
        check(graph['estate_id'] == estate)
        nodes = {(n['type'], n['id']): n['properties'] for n in graph['nodes']}
        edges = {(r['source_type'], r['source_id'], r['relation'], r['target_type'], r['target_id'])
                 for r in graph['relations'] if r['estate_id'] == estate and r['properties']['estate_id'] == estate}

        def edge(st, si, rel, tt, ti):
            check((st, si, rel, tt, ti) in edges)

        def reviewed(kind, identity, version, decision):
            matching = [r for (t, _), r in nodes.items() if t == 'ReviewDecision'
                        and r.get('target_id') == identity and r.get('target_type') == kind
                        and r.get('review_purpose') == 'POC_DEMO' and r.get('decision') == decision
                        and r.get('target_version_hash') == version and r.get('estate_id') == estate
                        and r.get('actor_role') == 'HUMAN_REVIEWER']
            check(matching)
            for r in matching:
                edge('ReviewDecision', r['review_decision_id'], 'REVIEWS', kind, identity)
            return matching

        rows[estate] = []
        check(len(manifest['policy_ids'][estate]) == 50)
        for identity in manifest['policy_ids'][estate]:
            p = nodes[('Policy', identity)]
            check(p['estate_id'] == estate and p['review_purpose'] == 'POC_DEMO'
                  and p['review_status'] == 'APPROVED' and p['demo_reviewed_hash'] == p['version_hash'])
            reviews = reviewed('Policy', identity, p['version_hash'], 'APPROVED')
            edge('Policy', identity, 'APPLIES_TO', 'Estate', estate)
            edge('Policy', identity, 'GOVERNS', 'Service', p['service_id'])
            edge('Estate', estate, 'OFFERS', 'Service', p['service_id'])
            evidence = []
            for chunk_id in json.loads(p['chunk_ids_json']):
                c = nodes[('EvidenceChunk', chunk_id)]
                d = nodes[('Document', c['document_id'])]
                check(c['estate_id'] == d['estate_id'] == estate and c['text'])
                check(sha((root/d['source_path']).read_bytes()) == d['sha256'])
                edge('Policy', identity, 'SUPPORTED_BY', 'EvidenceChunk', chunk_id)
                edge('Document', d['document_id'], 'HAS_CHUNK', 'EvidenceChunk', chunk_id)
                edge('Estate', estate, 'HAS_DOCUMENT', 'Document', d['document_id'])
                reviews += reviewed('EvidenceChunk', chunk_id, c['version_hash'], 'VERIFIED')
                reviews += reviewed('Document', d['document_id'], d['version_hash'], 'VERIFIED')
                evidence.append({'chunk_id': chunk_id, 'document_id': d['document_id'],
                                 'file': d['source_path'], 'locator': json.loads(c['source_locator_json']),
                                 'excerpt': c['text'], 'document_version': d['sha256'],
                                 'chunk_version': c['version_hash']})
            check(evidence and p['statement'] and p['question'])
            conditions = json.loads(p['conditions_json'])
            check(conditions['applicable_to']['estate_id'] == estate)
            # Preserve qualifiers, not the original document's unexecuted promises.
            answer = p['statement'] + ('\n适用说明：' + p['scope_note'] if p.get('scope_note') else '')
            row = {'qa_id': identity, 'estate_id': estate, 'question': p['question'], 'answer': answer,
                   'service': p['service_code'], 'conditions': conditions, 'source_para': identity,
                   'source_file': p['source_file'], 'source_locator': p['source_locator_json'],
                   'approval_date': p['demo_valid_from'], 'origin': 'EXISTING_POC_REVIEWED_POLICY_FACT',
                   'valid_to': p.get('demo_valid_to'), 'fact': json.loads(p['fact_json'])}
            rows[estate].append(row)
            proofs[identity] = {'record_sha256': p['version_hash'], 'source_path': evidence[0]['file'],
                                'answer_paragraphs': [], 'raw_excerpt': p['source_excerpt'],
                                'workbook_row': None, 'evidence_refs': evidence, 'review_events': reviews}
            references[identity] = {'document_id': evidence[0]['document_id'],
                                    'existing_evidence_chunk_ids': [e['chunk_id'] for e in evidence],
                                    'policy_id': identity}
            sources[evidence[0]['file']] = {'path': evidence[0]['file'], 'sha256': evidence[0]['document_version']}
    receipt = {**manifest, 'packet_sha256': config['manifest_sha256'], 'records': proofs,
               'graph_references': references, 'sources_by_path': sources}
    return rows, receipt, config
