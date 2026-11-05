"""Tenant-bound retrieval and existing approval gates, followed by the global answer selector."""

import json
import re

from ..answer_selection import verified_selection
from ..domain import POCError
from ..faq_contract import KNOWLEDGE_SERVICES
from ..faq_knowledge import confirmed_faq, demo_faq
from ..knowledge import confirmed_policy, demo_policy
from ..response_policy import classify_message, fixed_reply, review_required_reply
from .contract import AnswerResult, Candidate
from .intent import SPECIFIC, QuestionScope, canonical_question, facets, subjects
from .selector import business_conditions, compose

ESTATE_NAMES = {"E001": ("烟台港城华府", "港城华府"), "E002": ("青岛国际健康生态谷", "国际健康生态谷", "国际生态谷")}


def source_ref(entry):
    c, d = entry["chunk"], entry["document"]
    return {"document_id": d["document_id"], "chunk_id": c["chunk_id"], "file": d["source_path"],
            "locator": json.loads(c["source_locator_json"]), "excerpt": c["text"],
            "document_version": d.get("version_hash"), "chunk_version": c.get("version_hash")}


def subject_context(bundle, record):
    # Recover an omitted contact role from the actual evidence question, not from unrelated answer text.
    if record["topic"] != "contact" or subjects(record["title"] + record["question"]) & SPECIFIC:
        return ""
    questions = []
    for proof in bundle["evidence"]:
        text = proof["chunk"]["text"][:250]
        if "？" in text or "?" in text:
            questions.append(re.split(r"[？?]", text, 1)[0])
    return " ".join(questions)


def missing_context(record, unit, building, activity, *, faq=False):
    conditions = record["applicable_to"] if faq else record["conditions"]["applicable_to"]
    known = {**(unit or {}), "building_form": (building or {}).get("building_form"), "renovation_activity": activity}
    return [{"field": key, "required_value": value} for key, value in conditions.items()
            if key in {"residence_category", "has_elevator", "building_form", "renovation_activity"}
            and value not in (None, "UNSPECIFIED") and known.get(key) is None]


def policy_candidate(bundle, record, mode, score):
    p = record
    text = p["title"] + " " + p["question"]
    sources = [source_ref(e) for e in bundle["evidence"]]
    match = {"policy_id": p["policy_id"], "title": p["title"], "statement": p["statement"],
             "conditions": p["conditions"], "topic": p["topic"], "sources": sources,
             "usage": "POC_DEMO_ONLY" if mode == "demo" else "CURRENT_POLICY"}
    stored = bundle["policy"]
    window = tuple(stored.get(k) for k in (("demo_valid_from", "demo_valid_to") if mode == "demo" else ("valid_from", "valid_to")))
    return Candidate(p["policy_id"], p["estate_id"], p["version_hash"], p["title"], p["question"], p["statement"],
                     p["service_code"], "POLICY", facets(text, topic=p["topic"], fact=p["fact"]), subjects(text + subject_context(bundle, p)),
                     p["conditions"]["applicable_to"], window, p["fact"], sources,
                     notes=[p["conditions"]["description"]], public_conditions=business_conditions(p["conditions"]["description"]),
                     match=match, score=score, audit={"approved_mode": mode, "blocking_issues": p["blocking_issues"],
                         "source_record": p, "review_events": bundle["reviews"],
                         "ambiguous": any(q.startswith("AMBIGUOUS_") for q in p["conditions"]["qualifiers"])})


def faq_candidate(bundle, faq, mode, score, selection=None):
    f = bundle["faq"]
    notes = [selection["review_notes"]] if selection else []
    if "CALLBACK_OR_WORKORDER_PROMISE_REWRITE_REQUIRED" in faq["blocking_issues"]:
        notes.append("当前应用尚未实现咨询工单和自动回电，不代表已提交或已安排专人联系。")
    if faq["scope_status"] == "PROJECT_DETAILS_UNCONFIRMED":
        notes.append("具体项目参数、电话及日期的现行适用性仍需核验。")
    text = selection["selected_answer"] if selection else faq["answer_candidate"] if mode == "demo" else faq["reviewed_answer"]
    refs = [{"document_id": faq["document_id"], "chunk_id": faq["chunk_id"], "file": faq["source_file"],
             "locator": faq["source_locator"], "excerpt": faq["source_answer"], "document_version": faq["source_sha256"]}]
    match = {"faq_id": faq["faq_id"], "title": faq["question"], "statement": text,
             "conditions": {"description": selection["selected_conditions"] if selection else faq["human_review"]["confirmed_conditions"]},
             "topic": "faq", "sources": refs, "application_notes": notes,
             "usage": "POC_DEMO_ONLY" if mode == "demo" else "CURRENT_POLICY"}
    if selection:
        match.update(answer_source=selection["selected_source"], answer_selection_hash=selection["selection_hash"],
                     requires_context_clarification=selection["has_distinct_answers"])
    return Candidate(faq["faq_id"], faq["estate_id"], faq["version_hash"], faq["question"], faq["question"], text,
                     faq["service_code"], "FAQ", facets(faq["question"]), subjects(faq["question"]), faq["applicable_to"],
                     tuple(f.get(k) for k in (("demo_valid_from", "demo_valid_to") if mode == "demo" else ("valid_from", "valid_to"))),
                     sources=refs, notes=notes, match=match, score=score,
                     audit={"approved_mode": mode, "blocking_issues": faq["blocking_issues"], "scope_status": faq["scope_status"],
                            "source_record": faq, "review_events": bundle["reviews"],
                            "answer_selection": selection})


class KnowledgeAnswers:
    def __init__(self, repo, auth, root, day):
        self.repo, self.auth, self.root, self.day = repo, auth, root, day

    def answer(self, who, question, *, service_code=None, unit_id=None, knowledge_mode="current", renovation_activity=None,
               semantic_matcher=None, semantic_context=None, _capture=None, _recall_limit=12, _collect_all=False):
        if not isinstance(question, str) or not question.strip() or len(question) > 300:
            raise POCError("INVALID_INPUT", "问题需为1至300字", 400)
        from . import day1_poc
        if day1_poc.configuration(self.root) is not None:
            if knowledge_mode not in {"current", "demo"}:
                raise POCError("INVALID_INPUT", "知识用途无效", 400)
            return day1_poc.answer(self.root, self.repo, who, question.strip(), day=self.day,
                knowledge_mode=knowledge_mode, unit_id=unit_id, semantic_matcher=semantic_matcher,
                semantic_context=semantic_context)
        reason = classify_message(question)
        if reason:
            old = fixed_reply(who.estate_id, reason)
            return {**AnswerResult(status="NOT_FOUND", answer_text=old["message"], estate_id=who.estate_id,
                                   debug_info={"routing": reason}).model_dump(), **old}
        if service_code is not None and service_code not in KNOWLEDGE_SERVICES:
            raise POCError("INVALID_INPUT", "服务类别无效", 400)
        if knowledge_mode not in {"current", "demo"} or renovation_activity not in {None, "BARE", "MINOR", "DEMOLITION", "FLOOR_HEATING"}:
            raise POCError("INVALID_INPUT", "知识用途或装修活动无效", 400)
        question = question.strip()
        scope = QuestionScope.parse(question, who.estate_id, service_code)
        service_mismatch = bool(service_code and scope.inferred_service_code and service_code != scope.inferred_service_code)
        mentioned = set(re.findall(r"(?<![A-Z0-9])E00[12](?![A-Z0-9])", question.upper()))
        mentioned.update(e for e, names in ESTATE_NAMES.items() if any(n in question for n in names))
        unit = building = None
        if unit_id:
            eligible = self.repo.eligible_unit(who, unit_id, self.day, create=False)
            unit, building = eligible["unit"], eligible["building"]
        housing_mismatch = bool(unit and (("不带电梯" in question or "无电梯" in question) and unit.get("has_elevator") is True
            or "商铺" in question and unit.get("residence_category") != "SHOP"
            or "人才公寓" in question and unit.get("residence_category") != "TALENT_APARTMENT"))
        context = {"knowledge_mode": knowledge_mode, "scope_source": "VERIFIED_LOGIN", "unit_id": unit_id,
                   "renovation_activity": renovation_activity, "scope_mismatch": bool(mentioned - {who.estate_id}),
                   "housing_mismatch": housing_mismatch,
                   "verified_conditions": {k: (unit or {}).get(k) for k in ("residence_category", "has_elevator")},
                   "building_form": (building or {}).get("building_form")}
        candidates, rejected, missing, warnings = [], [], [], []
        if service_mismatch:
            warnings.append("SERVICE_HINT_MISMATCH_SEARCH_NOT_EXCLUDED")
        if context["scope_mismatch"] or housing_mismatch:
            return self._transport(compose(scope, [], context=context), [], knowledge_mode)
        bundles = self.repo.knowledge_bundles(who.estate_id)
        # Selection links may point into another service category, but never another estate.
        all_bundles = bundles
        records = {b["policy"]["policy_id"]: (b, json.loads(b["policy"]["_record_json"])) for b in all_bundles}
        faq_bundles = self.repo.faq_bundles(who.estate_id, knowledge_mode=knowledge_mode)
        semantic_enabled = True
        recalled = set()
        if semantic_enabled:
            from .templates import recall_ids
            descriptions = [(b["policy"]["policy_id"], p["question"], p["title"], p["service_code"])
                for b in bundles for p in [json.loads(b["policy"]["_record_json"])]]
            descriptions.extend((b["faq"]["faq_id"], f["question"], f["question"], f["service_code"])
                for b in faq_bundles for f in [json.loads(b["faq"]["_record_json"])])
            limit = semantic_matcher.match_settings.recall_groups if semantic_matcher else _recall_limit
            vectors = {}
            if not _collect_all and hasattr(self.repo, "semantic_index_rows"):
                from .semantic_index import recall
                try:
                    vectors, index_info = recall(self.root, who.estate_id, self.repo.semantic_index_rows(who.estate_id), question)
                except (OSError, ValueError, TypeError) as exc:
                    index_info = {"status": "UNAVAILABLE_INVALID_INDEX", "reason": type(exc).__name__, "approval_cache": False}
                    warnings.append("VECTOR_INDEX_UNAVAILABLE_USING_LIVE_SOURCE_RECALL")
                context["retrieval_index"] = index_info
            recalled = {r[0] for r in descriptions} if _collect_all else recall_ids(scope, descriptions, limit, vectors)
            # An ontology fact family (channels / duties) is a list of distinct
            # approved facts. Top-K must not silently drop its remaining members.
            families = {p["fact"]["fact_key"].rsplit(".", 1)[0] for _, p in records.values()
                if p["policy_id"] in recalled and p["fact"].get("predicate") in {"channel", "responsibility"}
                and scope.score(p["title"], p["question"], p["service_code"], p["topic"], p["fact"]) >= .6}
            recalled.update(p["policy_id"] for _, p in records.values()
                            if p["fact"].get("fact_key", "").rsplit(".", 1)[0] in families)
            context["expanded_fact_families"] = sorted(families)
        reader = demo_policy if knowledge_mode == "demo" else confirmed_policy
        checked = {}

        def check_policy(bundle, record):
            identity = record["policy_id"]
            if identity not in checked:
                checked[identity] = reader(bundle, who.estate_id, self.day, self.root, self.auth.reviewer_authorized,
                                           unit, building, renovation_activity)
            return checked[identity]

        for bundle in bundles:
            p = json.loads(bundle["policy"]["_record_json"])
            if p["estate_id"] != who.estate_id:
                warnings.append("OUT_OF_SCOPE_RECORD_REJECTED")
                continue
            score = scope.score(p["title"], p["question"], p["service_code"], p["topic"], p["fact"], subject_context(bundle, p))
            if semantic_enabled and p["policy_id"] not in recalled or not semantic_enabled and score < 0.6:
                continue
            if not check_policy(bundle, p):
                if score >= 0.6:
                    missing.extend({"id": p["policy_id"], **m} for m in missing_context(p, unit, building, renovation_activity))
                rejected.append({"id": p["policy_id"], "reason": "APPROVAL_VERSION_DATE_SCOPE_OR_EVIDENCE_GATE", "title": p["title"],
                                 "source_record": p, "review_events": bundle["reviews"], "evidence_refs": [source_ref(e) for e in bundle["evidence"]]})
                continue
            candidates.append(policy_candidate(bundle, p, knowledge_mode, score))
        faq_reader = demo_faq if knowledge_mode == "demo" else confirmed_faq
        raw_faqs = []
        for bundle in faq_bundles:
            f = json.loads(bundle["faq"]["_record_json"])
            if f["estate_id"] != who.estate_id:
                warnings.append("OUT_OF_SCOPE_RECORD_REJECTED")
                continue
            score = scope.score(f["question"], f["question"], f["service_code"])
            if semantic_enabled and f["faq_id"] in recalled or not semantic_enabled and score >= 0.6:
                raw_faqs.append((bundle, f, score))
        day1_groups = {canonical_question(f["question"]) for b, f, _ in raw_faqs
                       if knowledge_mode == "demo" and b["faq"].get("answer_selection_source") == "DAY1_POLICY"}
        blocked_ids, source_choice_failed = set(), False
        for bundle, f, score in raw_faqs:
            group = canonical_question(f["question"])
            stored = bundle["faq"]
            if knowledge_mode == "demo" and stored.get("answer_selection_json"):
                selection = verified_selection(bundle, who.estate_id, self.day, self.root, self.auth.reviewer_authorized,
                                               all_bundles, unit, building, renovation_activity)
                if not selection:
                    missing.extend({"id": f["faq_id"], **m} for m in missing_context(f, unit, building, renovation_activity, faq=True))
                    rejected.append({"id": f["faq_id"], "reason": "SOURCE_SELECTION_GATE", "title": f["question"],
                                     "source_record": f, "review_events": bundle["reviews"]})
                    source_choice_failed |= canonical_question(question) == group
                    try:
                        blocked_ids.update(p["policy_id"] for p in json.loads(stored["answer_selection_json"])["policy_versions"])
                    except (KeyError, ValueError, TypeError):
                        source_choice_failed |= score >= 0.9
                    continue
                if selection["selected_source"] == "DAY1_POLICY":
                    covered = False
                    for version in selection["policy_versions"]:
                        if version["policy_id"] not in records:
                            continue
                        b, p = records[version["policy_id"]]
                        policy_score = scope.score(p["title"], p["question"], p["service_code"], p["topic"], p["fact"], subject_context(b, p))
                        if not check_policy(b, p):
                            missing.extend({"id": p["policy_id"], **m} for m in missing_context(p, unit, building, renovation_activity))
                            continue
                        if not semantic_enabled and policy_score < 0.6:
                            continue
                        covered = True
                        c = policy_candidate(b, p, knowledge_mode, policy_score)
                        c.match.update(answer_source="DAY1_POLICY", faq_id=f["faq_id"], application_notes=[selection["review_notes"]])
                        c.audit["answer_selection"] = selection
                        c.audit.setdefault("binding_issues", {})[selection["selection_hash"]] = f.get("blocking_issues", [])
                        prior = next((x for x in candidates if x.identity == c.identity), None)
                        c.audit["question_variants"] = list(dict.fromkeys([
                            *(prior.audit.get("question_variants", []) if prior else []), selection["question"]]))
                        candidates = [x for x in candidates if x.identity != c.identity]
                        candidates.append(c)
                    if not covered:
                        rejected.append({"id": f["faq_id"], "reason": "DAY1_NOT_APPLICABLE_OR_NOT_ANSWERING_REQUEST", "selection": selection})
                    continue
                if group in day1_groups:
                    rejected.append({"id": f["faq_id"], "reason": "SUPERSEDED_BY_HUMAN_DAY1_CHOICE"})
                    continue
                candidates.append(faq_candidate(bundle, f, knowledge_mode, score, selection))
            else:
                if group in day1_groups:
                    rejected.append({"id": f["faq_id"], "reason": "SUPERSEDED_BY_HUMAN_DAY1_CHOICE"})
                    continue
                if faq_reader(bundle, who.estate_id, self.day, self.root, self.auth.reviewer_authorized, unit, building, renovation_activity):
                    candidates.append(faq_candidate(bundle, f, knowledge_mode, score))
                else:
                    missing.extend({"id": f["faq_id"], **m} for m in missing_context(f, unit, building, renovation_activity, faq=True))
                    rejected.append({"id": f["faq_id"], "reason": "FAQ_APPROVAL_VERSION_DATE_SCOPE_OR_EVIDENCE_GATE", "title": f["question"],
                                     "source_record": f, "review_events": bundle["reviews"]})
        if knowledge_mode == "demo":
            # Bind all existing, currently authorised source aliases to eligible
            # policies, independent of which FAQ happened to win Top-K today.
            available = {c.identity: c for c in candidates if c.origin == "POLICY"}
            for bundle in faq_bundles:
                stored = bundle["faq"]
                if stored.get("answer_selection_source") != "DAY1_POLICY":
                    continue
                try:
                    target_ids = {p["policy_id"] for p in json.loads(stored["answer_selection_json"])["policy_versions"]}
                except (KeyError, ValueError, TypeError):
                    continue
                if not target_ids.intersection(available):
                    continue
                selection = verified_selection(bundle, who.estate_id, self.day, self.root, self.auth.reviewer_authorized,
                                               all_bundles, unit, building, renovation_activity)
                if selection:
                    for identity in target_ids.intersection(available):
                        c = available[identity]
                        bindings = {r["selection_hash"]: r for r in c.audit.get("source_bindings", [])}
                        bindings[selection["selection_hash"]] = selection
                        c.audit["source_bindings"] = [bindings[k] for k in sorted(bindings)]
                        c.audit.setdefault("binding_issues", {})[selection["selection_hash"]] = json.loads(stored["_record_json"]).get("blocking_issues", [])
        if not candidates:
            for row in self.repo.pending_faq_questions(who.estate_id, service_code):
                if row.get("estate_id") != who.estate_id or not str(row.get("faq_id", "")).startswith(who.estate_id + ":FAQ:"):
                    continue
                if len(canonical_question(question)) >= 4 and scope.score(row["question"], row["question"], row.get("service_code") or service_code or "") >= 0.6:
                    rejected.append({"id": row["faq_id"], "title": row["question"], "reason": "PENDING_OR_UNAVAILABLE_FOR_MODE",
                                     "service_code": row.get("service_code"),
                                     "demo_content_confirmed": row.get("demo_review_status") == "APPROVED"})
        if source_choice_failed:
            rejected.extend({"id": c.identity, "reason": "AUTHORITATIVE_SOURCE_CHOICE_UNAVAILABLE"} for c in candidates)
            candidates = []
        elif blocked_ids:
            candidates = [c for c in candidates if c.identity not in blocked_ids]
        hard_quality = {"TRUNCATED_OR_PLACEHOLDER_CONTENT", "QUESTION_CONTEXT_MISSING",
                        "MULTIPLE_TOPICS_IN_ONE_CELL", "SAFETY_CLAIM_REQUIRES_QUALIFIED_REVIEW"}
        # Withholding an incomplete answer must not erase a relevant, already
        # observed disagreement. Keep the source records as conflict shadows;
        # they may block certainty but never supply resident-facing facts.
        comparison = compose(scope, [c for c in candidates if c.origin == "FAQ"
            and canonical_question(c.question) == canonical_question(question)], context=context)
        if comparison.status == "CONFLICT":
            context["unresolved_source_conflicts"] = comparison.debug_info["conflicts"]
            context["conflict_shadow_records"] = comparison.debug_info["candidates"]
        eligible = []
        for c in candidates:
            unresolved = hard_quality.intersection(c.audit.get("blocking_issues", [])) if c.origin == "FAQ" else set()
            if unresolved:
                rejected.append({"id": c.identity, "reason": "STRUCTURE_OR_QUALIFIED_REVIEW_REQUIRED",
                                 "issues": sorted(unresolved), "candidate": c.summary()})
            else:
                eligible.append(c)
        candidates = eligible
        answer = compose(scope, candidates, rejected=rejected, missing=missing, warnings=warnings, context=context)
        if context.get("unresolved_source_conflicts"):
            answer.debug_info["conflicts"] = context["unresolved_source_conflicts"]
            answer.debug_info["candidates"].extend(context["conflict_shadow_records"])
            answer.warnings.append("UNRESOLVED_CONFLICT")
        answer.debug_info["mode_notice"] = ("仅用于POC资料参考；演示可用起日不是真实政策生效日。" if knowledge_mode == "demo" else "正式政策模式")
        if _capture is not None:
            _capture.extend(candidates)
            return self._transport(answer, candidates, knowledge_mode)
        if semantic_matcher is None:
            from .templates import MatchSettings
            class LocalExactMatcher:
                match_settings = MatchSettings()
                def match_templates(self, *args, **kwargs):
                    return {"knowledge_ids": answer.debug_info.get("selected_ids", []) if answer.status == "ANSWERED" else [],
                            "sufficiency": "SUFFICIENT" if answer.status == "ANSWERED" else "NONE"}
            semantic_matcher = LocalExactMatcher()
        if semantic_matcher is not None:
            from .templates import TemplateAnswers
            def recheck():
                fresh = []
                self.answer(who, question, service_code=service_code, unit_id=unit_id, knowledge_mode=knowledge_mode,
                            renovation_activity=renovation_activity, _capture=fresh,
                            _recall_limit=semantic_matcher.match_settings.recall_groups)
                return fresh
            selected = TemplateAnswers(semantic_matcher).answer(scope, candidates, baseline=answer.model_dump(),
                context={**(semantic_context or {}), **context}, recheck=recheck)
            return self._transport(AnswerResult.model_validate(selected), candidates, knowledge_mode)
        return self._transport(answer, candidates, knowledge_mode)

    @staticmethod
    def _transport(answer, candidates, mode):
        """Legacy fields remain internal for existing diagnostic callers; residents see only answer_text."""
        result = answer.model_dump()
        if answer.status != "ANSWERED" and result["answer_source"] is None:
            result["answer_source"] = "GENERAL_CONVERSATION" if answer.status == "NOT_FOUND" else "CLARIFICATION"
        matches = result["debug_info"].get("raw_matches", [])
        if answer.status == "NOT_FOUND":
            legacy = fixed_reply(answer.estate_id, "NO_VERIFIED_MATCH")
        elif answer.status in {"NOT_VERIFIED", "NEED_CLARIFICATION"}:
            confirmed = any(r.get("demo_content_confirmed") for r in answer.debug_info.get("rejected", []))
            q = QuestionScope.parse(answer.debug_info.get("question", ""), answer.estate_id, answer.debug_info.get("explicit_service_code"))
            pending_faq = any(":FAQ:" in r.get("id", "") and q.score(r.get("title", ""), r.get("title", ""),
                r.get("source_record", {}).get("service_code") or r.get("service_code") or "") >= .6 for r in answer.debug_info.get("rejected", []))
            legacy = review_required_reply(answer.estate_id, mode, demo_content_confirmed=confirmed) if pending_faq else fixed_reply(answer.estate_id, "NO_VERIFIED_MATCH")
        else:
            legacy = {"response_type": "KNOWLEDGE", "message": answer.answer_text, "red_line": False}
        result.update(legacy)
        result["matches"] = matches
        result["knowledge_mode"] = "POC_DEMO" if mode == "demo" else "CURRENT_POLICY"
        # No candidate limit, textual metadata concatenation, or second answer generation step.
        return result
