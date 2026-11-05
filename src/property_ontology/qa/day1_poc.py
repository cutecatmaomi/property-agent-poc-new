"""Golden Questions entrypoint; historical module name retained for imports. No legacy graph retrieval or implicit approvals."""
import hashlib
import json
import re
from pathlib import Path

from ..domain import POCError
from .contract import AnswerResult
from .templates import SemanticDecision

SET = "GOLDEN_QUESTIONS_100"
FOLDER = Path("data/knowledge_sets") / SET
QUOTE_NOTICE = "以下是历史交付资料中的原答复引用。本次没有安排人员、预约上门或完成整改；需要报修时，请另行核对并确认提交。"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition):
    # Authorization checks must remain active even when Python uses -O.
    if not condition:
        raise ValueError('invalid approved packet')


def exact_key(text):
    # Never remove negation, numbers, subjects or conditions.
    return re.sub(r"[\s？?。！!，,；;]", "", text).casefold()


def configuration(root):
    path = Path(root) / "config/knowledge_source.json"
    if not path.exists():
        if ((Path(root)/FOLDER/'GoldenQuestions_100问答.json').exists()
                or (Path(root)/'data/knowledge_sets/ONTOLOGY_PACKAGE_POC/manifest.json').exists()):
            raise POCError('KNOWLEDGE_CONFIG_INVALID', '知识来源配置缺失，暂时不能引用资料。', 503)
        return None
    try:
        config = json.loads(path.read_text())
        if config['active_set'] != SET or config['legacy_fallback'] is not False:
            raise ValueError('unsupported source switch')
        return config
    except (ValueError, KeyError, TypeError, OSError):
        raise POCError('KNOWLEDGE_CONFIG_INVALID', '知识配置暂不可用，请联系维护人员核对。', 503) from None


def load(root):
    """Revalidate approval and source bytes on every lookup; vector cache is not authority."""
    root = Path(root)
    config = configuration(root)
    if config is None:
        return None
    try:
        from .golden_poc import load as load_golden
        return load_golden(root, config)
    except (OSError, ValueError, KeyError, TypeError, AssertionError):
        raise POCError('KNOWLEDGE_PACKET_INVALID', '这批资料的版本或授权需要重新核对，暂时不能引用。', 503) from None


def session_root(session):
    return getattr(getattr(session.actions, 'settings', None), 'root', None)


def knowledge_epoch(session):
    root = session_root(session)
    config = configuration(root) if root else None
    return config['manifest_sha256'] if config else None


def exact_question(session, message):
    root = session_root(session)
    packet = load(root) if root else None
    return bool(packet and any(exact_key(r['question']) == exact_key(message) for r in packet[0][session.estate_id]))


def projected(record, receipt):
    proof = receipt['records'][record['qa_id']]
    return {'template_id': record['qa_id'], 'estate_id': record['estate_id'],
        'canonical_question': record['question'], 'question_variants': [], 'standard_answer': record['answer'],
        'conditions': {'source_conditions': record['conditions'], 'question_scope': record['question'], 'mode': 'POC_ONLY'},
        'service_code': record['service'], 'priority': 100, 'source_type': record.get('source_type', 'ONTOLOGY_PACKAGE_POC'), 'source_priority': 100,
        'usage_mode': 'POC_ONLY', 'version_hash': proof['record_sha256']}


def answer(root, repo, who, question, *, day, knowledge_mode, unit_id=None, semantic_matcher=None, semantic_context=None):
    packet = load(root)
    if packet is None:
        raise ValueError('exclusive source is not configured')
    rows, receipt, config = packet
    active_set = config['active_set']
    debug = {'knowledge_set': active_set, 'dataset_hash': receipt['packet_sha256'], 'legacy_retrieval_called': False,
             'user_question': question, 'semantic_validation': None, 'generation_sufficiency': 'NONE'}

    def result(status, text, **extra):
        default_source = 'GENERAL_CONVERSATION' if status == 'NOT_FOUND' else 'CLARIFICATION'
        value = AnswerResult(status=status, answer_text=text, estate_id=who.estate_id, knowledge_set=active_set,
            answer_source=extra.pop('answer_source', default_source), debug_info=debug,
            response_mode=('TEMPLATE_GOLDEN_QUESTIONS' if active_set == SET else 'TEMPLATE_ONTOLOGY_POC') if status == 'ANSWERED' else status, **extra).model_dump()
        return {**value, 'message': text, 'matches': [], 'response_type': 'KNOWLEDGE' if status == 'ANSWERED' else 'REVIEW_REQUIRED'}

    if knowledge_mode != 'demo':
        return result('NOT_VERIFIED', '这批资料仅获准用于演示，当前有效政策尚未确认。', decision_reason='POC_ONLY')
    if str(day) < receipt['approval_date']:
        return result('NOT_VERIFIED', '这批资料尚未到演示可用日期。', decision_reason='BEFORE_POC_AUTHORIZATION')
    from .service import ESTATE_NAMES
    mentions = set(re.findall(r'E00[12]', question.upper()))
    mentions.update(e for e, names in ESTATE_NAMES.items() if any(n in question for n in names))
    if mentions - {who.estate_id}:
        return result('NEED_CLARIFICATION', '请切换到要咨询的楼盘账号后再查询。', decision_reason='ESTATE_MISMATCH')
    unit = building = {}
    if unit_id:
        eligible = repo.eligible_unit(who, unit_id, day, create=False)
        unit, building = eligible['unit'], eligible['building']
    candidates = rows[who.estate_id]
    # An LLM rewrite is a search hint, not a literal user question. It cannot
    # bypass semantic coverage checks by rewriting an unsafe narrower question.
    actual_question = (semantic_context or {}).get('user_question') or question
    exact = [r for r in candidates if exact_key(r['question']) == exact_key(actual_question)]
    selected = exact[0] if len(exact) == 1 else None
    score = 1.0 if selected else None
    reason = 'EXACT_APPROVED_QUESTION' if selected else 'NO_ELIGIBLE_TEMPLATE'
    if not selected and semantic_matcher is not None:
        # A valid joint decision already examined ALL 50 current templates.
        # Validate its binding first; ranking the same full catalog again adds
        # work without adding candidates, evidence or a new authorization check.
        templates = [projected(r, receipt) for r in candidates]
        context = {**(semantic_context or {}), 'estate_id': who.estate_id, 'knowledge_set': active_set,
            'verified_conditions': ({k: unit.get(k) for k in ('has_elevator', 'residence_category')}
                                    if re.search(r'我家|我的房|我住|我们家', question) else {}),
            'building_form': building.get('building_form') if '我家' in question else None}
        # The planner's expanded query is a recall aid, not permission to invent
        # extra requested facts (materials, deadlines, fees, etc.). Resolve short
        # follow-ups with history, but assess coverage against the actual utterance.
        match_question = context.get('user_question') or question
        context['resolved_query'] = question
        hints = context.pop('_joint_selections', [])
        debug['joint_planning'] = context.pop('_planning_diagnostics', {})
        from ..agent.knowledge_planning import BoundSelection
        reused = [d for hint in hints if isinstance(hint, BoundSelection)
                  and (d := hint.reuse(match_question, context, templates)) is not None]
        debug['user_question'] = match_question
        try:
            if len(reused) == 1:
                decision = reused[0]
                debug['semantic_request_strategy'] = 'JOINT_WITH_PLANNING'
                index_info = {'status': 'JOINT_VALIDATED_FULL_CATALOG', 'records': len(templates),
                              'approval_cache': False}
            else:
                from .semantic_index import recall
                index_rows = [{'id': r['estate_id'] + ':GOLDEN:' + r['qa_id'], 'estate_id': r['estate_id'],
                    'question': r['question'], 'title': r['question'], 'service_code': r['service'],
                    'version_hash': receipt['records'][r['qa_id']]['record_sha256'], 'concepts': [],
                    'source_type': r.get('source_type', 'ONTOLOGY_PACKAGE_POC'), 'knowledge_set': active_set} for r in candidates]
                scores, index_info = recall(root, who.estate_id, index_rows, question, namespace=active_set)
                # Keep every candidate, including colloquial and low lexical
                # overlap questions. Ranking is an aid, never an answer gate.
                ranked = sorted(candidates, key=lambda r: scores.get(who.estate_id + ':GOLDEN:' + r['qa_id'], 0), reverse=True)
                templates = [projected(r, receipt) for r in ranked]
                debug['semantic_request_strategy'] = 'SEPARATE'
                debug['joint_fallback_reason'] = ('BINDING_OR_VERIFIED_CONDITIONS_CHANGED' if hints
                    else 'NO_VALID_PLANNING_SELECTION')
                decision = SemanticDecision.model_validate(semantic_matcher.match_templates(match_question, context=context, templates=templates, knowledge=[]))
        except POCError as exc:
            if exc.code != 'INVALID_TEMPLATE_MATCH':
                raise
            debug['semantic_protocol_retry'] = getattr(exc, 'diagnostics', {'reason': exc.code})
            decision = SemanticDecision.model_validate(semantic_matcher.match_templates(match_question, context={**context,
                'protocol_retry': '上次结构化输出格式错误。只修复JSON协议；返回合法枚举，禁止null、列表外ID或改动版本。不要生成答案。'},
                templates=templates, knowledge=[]))
        debug.update(retrieval_index=index_info, semantic_validation=decision.model_dump(), candidate_count=len(templates))
        offered = {t['template_id']: t for t in templates}
        if any(m.template_id not in offered or m.version_hash != offered[m.template_id]['version_hash'] for m in decision.matches):
            return result('NOT_VERIFIED', '这项资料的匹配需要重新核对。', decision_reason='INVALID_MATCH_REFERENCE')
        if decision.conflict_template_ids:
            return result('CONFLICT', '这项信息存在不同说法，需要先向物业核实。', decision_reason='CONFLICT')
        eligible = [m for m in decision.matches if m.coverage == 'FULL' and not m.additional_fact_required
                    and m.question_relation in {'EQUIVALENT', 'NARROWER'}]
        # Scores alone never authorize a template. Equivalence and full coverage
        # are mandatory; calibration, when present, is bound to dataset and model.
        calibration = config.get('calibration', {})
        model = getattr(getattr(semantic_matcher, 'settings', None), 'model', None)
        if calibration.get('dataset_hash') == receipt['packet_sha256'] and calibration.get('model') == model:
            eligible = [m for m in eligible if m.score >= calibration['threshold']]
        eligible.sort(key=lambda m: (m.question_relation == 'EQUIVALENT', m.score), reverse=True)
        if eligible:
            best = eligible[0]
            # Related full/split answers from the SAME original Q&A are compatible.
            # Distinct source Q&As require a clear semantic separation, never first-row wins.
            competing = [m for m in eligible[1:] if m.question_relation == best.question_relation
                         and best.score - m.score < calibration.get('margin', .05)]
            records = {r['qa_id']: r for r in candidates}
            different = [m for m in competing if records[m.template_id]['source_para'] != records[best.template_id]['source_para']
                         and records[m.template_id]['answer'] != records[best.template_id]['answer']]
            if different:
                return result('NEED_CLARIFICATION', '你具体想了解哪一项服务或哪种房屋的规定？', decision_reason='AMBIGUOUS_TEMPLATES')
            selected, score, reason = records[best.template_id], best.score, 'SEMANTIC_FULL_COVERAGE'
        elif decision.clarification != 'NONE':
            prompt = {'HOUSING': '你想了解哪种房屋类型的标准？', 'CONTACT_MODE': '你想了解现场服务时间，还是电话联系时间？'}
            options = list(dict.fromkeys(offered[m.template_id]['canonical_question'] for m in decision.matches if m.coverage == 'PARTIAL'))[:2]
            clarification = ('你具体想了解哪一点：' + '；'.join(options)) if options else '你具体想了解哪个部位或哪项服务？'
            return result('NEED_CLARIFICATION', prompt.get(decision.clarification, clarification), decision_reason=decision.clarification)
    if not selected:
        return result('NOT_FOUND', '这项具体信息目前没有可核实的答复。', decision_reason=reason)
    if selected.get('valid_to') and str(day) > selected['valid_to']:
        return result('NOT_VERIFIED', '这项资料已超出演示授权期限，需要重新核实。', decision_reason='POC_AUTH_EXPIRED')
    proof = receipt['records'][selected['qa_id']]
    source = receipt['sources_by_path'][proof['source_path']]
    references = receipt['graph_references'][selected['qa_id']]
    # Always label operational original wording as a quotation, never as a tool receipt.
    historical = bool(re.search(r'安排|整改|今天|马上|预计|我会|我们会|我司会|记录下来|联系维修|联系专业', selected['answer']))
    debug.update(selected_question=selected['question'], source_type=selected.get('source_type', 'ONTOLOGY_PACKAGE_POC'), template_eligibility='POC_APPROVED',
                 conditions=selected['conditions'], generation_sufficiency='SUFFICIENT', operation_executed=False,
                 graph_reference_mapping=references, answer_origin=selected.get('origin', 'ORIGINAL_QA'),
                 review_events=proof.get('review_events', []), question_set_id=config.get('question_set_id'),
                 golden_question_id=selected.get('question_id'))
    return result('ANSWERED', selected['answer'], answer_source='TEMPLATE', standard_answer=selected['answer'],
        matched_template_id=selected['qa_id'], template_version_hash=proof['record_sha256'], match_score=score,
        source_type=selected.get('source_type', 'ONTOLOGY_PACKAGE_POC'), review_status='POC_APPROVED', decision_reason=reason,
        usage_authorization={'scope': 'POC_ONLY', 'date': selected['approval_date'], 'policy_review_status': 'PENDING_REVIEW'},
        matched_policy_ids=[references['policy_id']] if references.get('policy_id') else [],
        evidence_refs=proof.get('evidence_refs') or [{'document_id': references['document_id'],
            'packet_evidence_id': selected['qa_id'], 'existing_chunk_ids': references['existing_evidence_chunk_ids'], 'file': source['path'],
            'original_supplied_filename': selected['source_file'], 'locator': selected['source_locator'],
            'paragraphs': proof['answer_paragraphs'], 'excerpt': proof['raw_excerpt'], 'document_version': source['sha256'],
            'workbook_row': proof['workbook_row']}],
        quote_notice=QUOTE_NOTICE if historical else None,
        warnings=['POC_ONLY_CURRENT_VALIDITY_UNCONFIRMED'] + (['HISTORICAL_OPERATION_QUOTATION'] if historical else []))
