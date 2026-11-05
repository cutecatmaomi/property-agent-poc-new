"""Template projections and semantic selection over freshly gated existing knowledge, not new graph nodes."""

import copy
import json
import re
from typing import Literal

from pydantic import Field

from ..domain import POCError, Strict, digest
from .contract import AnswerResult
from .intent import SPECIFIC, canonical_question, content_key, facet_matches, facets, grams, subjects
from .selector import deduplicate, unique, usable_text
from .template_catalog import Template as Template
from .template_catalog import project_templates as project_templates

MATCHING_REVISION = "source-priority-v6"


class TemplateMatch(Strict):
    template_id: str
    version_hash: str
    coverage: Literal["FULL", "PARTIAL", "NONE"]
    score: float = Field(ge=0, le=1)
    reason: str = Field(max_length=300)
    question_relation: Literal["EQUIVALENT", "NARROWER", "RELATED", "DIFFERENT", "UNSPECIFIED"] = "UNSPECIFIED"
    additional_fact_required: bool = True


class SourceComparison(Strict):
    left: str
    right: str
    relation: Literal["EQUIVALENT", "COMPLEMENTARY", "CONFLICT", "UNRELATED"]


class SemanticDecision(Strict):
    matches: list[TemplateMatch] = Field(default_factory=list, max_length=30)
    conflict_template_ids: list[str] = Field(default_factory=list, max_length=12)
    other_fact_conflict_ids: list[str] = Field(default_factory=list, max_length=12)
    knowledge_ids: list[str] = Field(default_factory=list, max_length=32)
    sufficiency: Literal["SUFFICIENT", "PARTIAL", "NONE"] = "NONE"
    clarification: Literal["NONE", "CONTACT_MODE", "HOUSING", "SUBJECT", "FACT"] = "NONE"
    reason: str = Field(default="", max_length=400)
    source_comparisons: list[SourceComparison] = Field(default_factory=list, max_length=12)


class MatchBoundary(Strict):
    threshold: float = Field(ge=0, le=1)
    margin: float = Field(ge=0, le=1)


class MatchSettings(Strict):
    recall_groups: int = Field(default=12, ge=4, le=30)
    threshold: float | None = Field(default=None, ge=0, le=1)
    margin: float | None = Field(default=None, ge=0, le=1)
    calibration_hash: str | None = None
    calibration_model: str | None = None
    calibration_revision: str | None = None
    by_service: dict[str, MatchBoundary] = Field(default_factory=dict)

    def boundary(self, service):
        measured = self.by_service.get(service)
        return (measured.threshold, measured.margin) if measured else (self.threshold, self.margin)

    @classmethod
    def load(cls, root):
        path = root / "config/template_matching.json"
        return cls.model_validate(json.loads(path.read_text())) if path.exists() else cls()


def recall_ids(scope, rows, limit, vector_scores=None):
    """Lexical broad recall of an LLM-resolved query; never described as an embedding score."""
    grouped = {}
    for identity, question, title, service in rows:
        score = scope.score(title, question, service)
        overlap = len(grams(scope.question) & grams(question)) / max(1, len(grams(scope.question)))
        key = (service, canonical_question(question))
        group = grouped.setdefault(key, {"rank": (score, overlap), "ids": [], "vector": -1})
        group["rank"] = max(group["rank"], (score, overlap))
        group["ids"].append(identity)
        group["vector"] = max(group["vector"], (vector_scores or {}).get(identity, -1))
    ranked = sorted(grouped.values(), key=lambda g: g["rank"], reverse=True)
    selected = [g for g in ranked if g["rank"][0] == 1]
    quota = max(1, limit // 3)
    for token in (":POL:", ":FAQ:"):
        selected.extend([g for g in ranked if any(token in i for i in g["ids"])][:quota])
    selected.extend(sorted(grouped.values(), key=lambda g: g["vector"], reverse=True)[:quota] if vector_scores else [])
    selected = list({id(g): g for g in selected}.values())
    selected.extend(g for g in ranked if g not in selected and len(selected) < limit)
    return {i for group in selected for i in group["ids"]}


def calibrate(observations, model):
    """Select observed score/margin cutoffs with zero false accepts, maximising retained positives."""
    # Use only actually observed eligible boundaries, never extrapolate acceptance to an untested zero score.
    scores = sorted({r["score"] for r in observations if r["eligible"]})
    margins = sorted({r["margin"] for r in observations if r["eligible"]})
    options = []
    for threshold in scores:
        for margin in margins:
            accepted = [r for r in observations if r["eligible"] and r["score"] >= threshold and r["margin"] >= margin]
            false = sum(not r["correct"] for r in accepted)
            true = sum(r["correct"] for r in accepted)
            if not false and true:
                options.append((true, threshold, margin))
    if not options:
        return MatchSettings(calibration_hash=digest(observations), calibration_model=model, calibration_revision=MATCHING_REVISION)
    _, threshold, margin = max(options, key=lambda o: (o[0], -o[1], -o[2]))
    by_service = {}
    for service in sorted({r.get("service_code") for r in observations if r.get("service_code")}):
        subset = [r for r in observations if r.get("service_code") == service]
        if any(r["correct"] for r in subset) and any(not r["correct"] for r in subset):
            # Fit the same zero-false-accept rule to this measured business
            # stratum. Removing service_code makes recursion terminate.
            fit = calibrate([{k: v for k, v in r.items() if k != "service_code"} for r in subset], model)
            if fit.threshold is not None:
                by_service[service] = MatchBoundary(threshold=fit.threshold, margin=fit.margin)
    return MatchSettings(threshold=threshold, margin=margin, calibration_hash=digest(observations), calibration_model=model,
                         calibration_revision=MATCHING_REVISION, by_service=by_service)


def scalar_disagreements(selected, templates, scope):
    dimensions = {"price": r"\d+(?:\.\d+)?\s*元\s*[/／]\s*(?:平方米|㎡|平米|户)",
                  "hours": r"\d{1,2}[:：]\d{2}\s*[—–至-]\s*\d{1,2}[:：]\d{2}"}
    subject = subjects(selected.canonical_question) & SPECIFIC
    conflicts = []
    for other in templates:
        if other.source_type == selected.source_type or other.service_code != selected.service_code:
            continue
        if not subject or subject != subjects(other.canonical_question) & SPECIFIC:
            continue
        for dimension, pattern in dimensions.items():
            if dimension not in scope.facets and dimension not in facets(selected.canonical_question):
                continue
            a = {re.sub(r"\s|／", "", x) for x in re.findall(pattern, selected.standard_answer)}
            b = {re.sub(r"\s|／", "", x) for x in re.findall(pattern, other.standard_answer)}
            if a and b and a != b:
                conflicts.append(other)
    return conflicts


class TemplateAnswers:
    def __init__(self, matcher):
        self.matcher = matcher
        self.settings = matcher.match_settings

    def answer(self, scope, candidates, *, baseline, context, recheck):
        if scope.estate_id is None:
            return {**baseline, "status": "NEED_CLARIFICATION", "answer_source": "CLARIFICATION",
                    "answer_text": "请问您咨询的是烟台港城华府，还是青岛国际健康生态谷？",
                    "matched_template_id": None, "matched_policy_ids": [], "evidence_refs": [],
                    "decision_reason": "ESTATE_REQUIRED"}
        def in_estate(c):
            return (c.estate_id == scope.estate_id and c.identity.startswith(scope.estate_id + ":")
                    and bool(c.sources) and all(ref.get("document_id", "").startswith(scope.estate_id + ":")
                    and (not ref.get("chunk_id") or ref["chunk_id"].startswith(scope.estate_id + ":")) for ref in c.sources))

        foreign = [c.identity for c in candidates if not in_estate(c)]
        candidates = [c for c in candidates if in_estate(c)]
        candidates, duplicates, identity_conflicts = deduplicate(candidates)
        templates, excluded = project_templates(candidates)
        audit = {**copy.deepcopy(baseline["debug_info"]), "templates": [t.model_dump() for t in templates],
                 "template_exclusions": excluded, "template_duplicates": duplicates, "out_of_scope_candidates": foreign,
                 "lexical_scores_are_semantic": False, "matching_settings": self.settings.model_dump()}

        def result(status, text, source, *, chosen=None, score=None, reason="", facts=None):
            refs = chosen.evidence_refs if chosen else unique([s for c in facts or [] for s in c.sources])
            ids = chosen.linked_policy_ids if chosen else [c.identity for c in facts or [] if c.origin == "POLICY"]
            selected_rows = [c for c in candidates if c.identity in chosen.source_ids] if chosen else facts or []
            if chosen:
                audit["preselection_raw_matches"] = audit.get("raw_matches", [])
                audit["raw_matches"] = [{**c.match, "sources": c.sources,
                    "answer_source": "DAY1_POLICY" if chosen.source_type == "DAY1" else "SUPPLEMENT_QA"} for c in selected_rows]
                audit["selected_ids"] = chosen.source_ids
            warnings = list(dict.fromkeys([*baseline["warnings"], *[note for c in selected_rows for note in
                [*c.notes, *c.audit.get("blocking_issues", []), *usable_text(c)[1]]]]))
            return AnswerResult(status=status, answer_text=text, answer_source=source,
                matched_template_id=chosen.template_id if chosen else None, match_score=score,
                matched_policy_ids=ids, evidence_refs=refs, estate_id=scope.estate_id,
                review_status=chosen.review_status if chosen else None, decision_reason=reason,
                warnings=warnings, debug_info=audit,
                source_type=chosen.source_type if chosen else None,
                response_mode=("TEMPLATE_" + chosen.source_type) if chosen else "NOT_VERIFIED" if status == "NOT_VERIFIED" else source,
                usage_authorization=chosen.usage_authorization if chosen else {},
                equivalence_group_id=chosen.equivalence_group_id if chosen else None,
                template_version_hash=chosen.version_hash if chosen else None,
                standard_answer=chosen.standard_answer if chosen else None).model_dump()

        # Existing scope/house checks are authoritative, even if a model asks to override them.
        if context.get("scope_mismatch") or context.get("housing_mismatch"):
            return result("NEED_CLARIFICATION", baseline["answer_text"], "CLARIFICATION", reason="SCOPE_OR_HOUSING_MISMATCH")
        if baseline["status"] == "NEED_CLARIFICATION" and baseline["debug_info"].get("missing_context"):
            return result("NEED_CLARIFICATION", baseline["answer_text"], "CLARIFICATION", reason="REQUIRED_BUSINESS_CONDITION_MISSING")
        if context.get("unresolved_source_conflicts"):
            audit["template_conflicts"] = context["unresolved_source_conflicts"]
            return result("CONFLICT", "相关资料有不同记载，目前无法确认，请联系物业核实。", "CLARIFICATION", reason="UNRESOLVED_SOURCE_DISAGREEMENT")
        if identity_conflicts:
            audit["template_conflicts"] = identity_conflicts
            return result("CONFLICT", "相关答复存在不一致，目前无法确认，请联系物业核实。", "CLARIFICATION", reason="ID_CONTENT_CONFLICT")
        if not candidates:
            return result(baseline["status"], baseline["answer_text"], "CLARIFICATION" if baseline["status"] != "NOT_FOUND" else "GENERAL_CONVERSATION", reason="NO_ELIGIBLE_KNOWLEDGE")
        raw_question = context.get("user_question") or scope.question
        exact = [t for t in templates if canonical_question(raw_question) in {canonical_question(t.canonical_question), *map(canonical_question, t.question_variants)}]
        cross_source_check = bool(exact and getattr(self.matcher, "semantic_available", False)
                                 and len({t.source_type for t in templates}) > 1)
        if exact and not cross_source_check:
            matches = [TemplateMatch(template_id=t.template_id, version_hash=t.version_hash, coverage="FULL", score=1,
                                     reason="EXACT_CANONICAL_QUESTION") for t in exact]
            decision = SemanticDecision(matches=matches, knowledge_ids=[i for t in exact for i in t.source_ids], sufficiency="SUFFICIENT")
            audit["semantic_validation"] = "EXACT_STANDARD_QUESTION"
        else:
            decision = SemanticDecision.model_validate(self.matcher.match_templates(raw_question, context=context,
                templates=[t.model_dump() for t in templates], knowledge=[{"id": c.identity, "question": c.question,
                "text": usable_text(c)[0], "conditions": c.scope} for c in candidates if usable_text(c)[0] and not c.audit.get("ambiguous")]))
            audit["semantic_validation"] = "LLM_STRUCTURED_COVERAGE"
            if exact:
                audit["model_semantic_decision"] = decision.model_dump()
                kept = [m for m in decision.matches if m.template_id not in {t.template_id for t in exact}]
                decision = decision.model_copy(update={"matches": [*kept, *[TemplateMatch(template_id=t.template_id,
                    version_hash=t.version_hash, coverage="FULL", score=1, reason="EXACT_REVIEWED_PAIR",
                    question_relation="EQUIVALENT", additional_fact_required=False) for t in exact]]})
                audit["semantic_validation"] = "EXACT_PAIR_WITH_CROSS_SOURCE_CHECK"
        audit["semantic_decision"] = decision.model_dump()
        known = {t.template_id: t for t in templates}
        rows = {c.identity: c for c in candidates}
        invalid_matches = [m.model_dump() for m in decision.matches if m.template_id not in known or m.version_hash != known[m.template_id].version_hash]
        invalid_knowledge = [i for i in decision.knowledge_ids if i not in rows]
        invalid_conflicts = [i for i in [*decision.conflict_template_ids, *decision.other_fact_conflict_ids] if i not in known]
        if invalid_matches or invalid_knowledge or invalid_conflicts:
            error = POCError("INVALID_TEMPLATE_MATCH", "暂时无法准确核对这项资料，请稍后再试。", 503)
            error.diagnostics = {**audit, "invalid_matches": invalid_matches, "invalid_knowledge_ids": invalid_knowledge,
                                 "invalid_conflict_ids": invalid_conflicts}
            raise error
        for comparison in decision.source_comparisons:
            if comparison.left not in known or comparison.right not in known:
                raise POCError("INVALID_TEMPLATE_MATCH", "暂时无法准确核对来源关系，请稍后再试。", 503)
        # A reviewed question/answer pair certifies the answer to its canonical
        # question. Equivalent questions with no new facts do not require a model
        # to approve the answer text a second time (or rewrite it).
        effective_matches = [m.model_copy(update={"coverage": "FULL"}) if m.question_relation == "EQUIVALENT"
            and not m.additional_fact_required else m for m in decision.matches]
        audit["effective_template_matches"] = [m.model_dump() for m in effective_matches]
        full = sorted([m for m in effective_matches if m.coverage == "FULL"], key=lambda m: m.score, reverse=True)
        # A typed boolean fact cannot be contradicted merely by disagreement over
        # another scalar (e.g. two different prices). Keep those conflicts audited.
        # A competing FULL answer or the same fact key remains a blocking conflict.
        nonblocking = False
        if full and decision.conflict_template_ids:
            target = known[full[0].template_id]
            target_rows = [rows[i] for i in target.source_ids]
            match_by_id = {m.template_id: m for m in decision.matches}
            conflicting_rows = [rows[i] for t in decision.conflict_template_ids for i in known[t].source_ids]
            keys = {c.fact.get("fact_key") for c in target_rows}
            nonblocking = (len(target_rows) == 1 and isinstance(target_rows[0].fact.get("value"), bool)
                and all(t in match_by_id and match_by_id[t].coverage == "PARTIAL" for t in decision.conflict_template_ids)
                and any(c.origin == "POLICY" and c.fact.get("fact_key") not in keys for c in conflicting_rows)
                and all(c.origin != "POLICY" or c.fact.get("fact_key") not in keys and not isinstance(c.fact.get("value"), bool)
                        for c in conflicting_rows))
            if nonblocking:
                audit["nonblocking_fact_conflicts"] = [known[i].model_dump() for i in decision.conflict_template_ids]
        if decision.conflict_template_ids and not nonblocking:
            audit["template_conflicts"] = [known[i].model_dump() for i in decision.conflict_template_ids]
            return result("CONFLICT", "这项资料有不同答复，目前无法确认，请联系物业核实。", "CLARIFICATION", reason="SEMANTIC_FACT_CONFLICT")
        if decision.clarification != "NONE":
            texts = {"CONTACT_MODE": "您想问现场服务时间，还是电话联系时间？", "HOUSING": "请确认您咨询的房屋类型。",
                     "SUBJECT": "您具体想了解哪个服务或部位？", "FACT": "您想了解哪一方面的信息？"}
            return result("NEED_CLARIFICATION", texts[decision.clarification], "CLARIFICATION", reason="SEMANTIC_CONTEXT_REQUIRED")
        if full:
            calibrated = self.settings.threshold is not None and self.settings.margin is not None
            qualified = [m for m in full if exact or calibrated and m.score >= self.settings.boundary(known[m.template_id].service_code)[0]]
            if not qualified:
                audit["template_top"] = {"id": full[0].template_id, "score": full[0].score, "margin": 1.0, "eligible": True}
                return result("NEED_CLARIFICATION", "请补充具体事项或适用情况，以便确认准确的标准答复。", "CLARIFICATION", reason="TEMPLATE_MATCH_NOT_CONFIDENT_NO_REWRITE")
            top = max(qualified, key=lambda m: (known[m.template_id].source_priority, m.score))
            selected = known[top.template_id]
            threshold, required_margin = self.settings.boundary(selected.service_code)
            # An explicit human Day1 source binding can supersede its individual
            # component templates; unrelated different answers cannot use priority.
            supersets = [m for m in qualified if known[m.template_id].priority > selected.priority and
                         set(known[m.template_id].source_ids) >= set(selected.source_ids)]
            if supersets:
                top = supersets[0]
                selected = known[top.template_id]
            comparisons = {frozenset((r.left, r.right)): r.relation for r in decision.source_comparisons}
            for comparison in decision.source_comparisons:
                if comparison.relation == "CONFLICT" and selected.template_id in {comparison.left, comparison.right}:
                    audit["template_conflicts"] = [known[i].model_dump() for i in (comparison.left, comparison.right)]
                    return result("CONFLICT", "这项资料有不同答复，目前无法确认，请联系物业核实。", "CLARIFICATION", reason="CROSS_SOURCE_FACT_CONFLICT")
            if selected.source_type == "ADDITIONAL_QA" and any(known[m.template_id].source_type == "DAY1" and
                    (comparisons.get(frozenset((selected.template_id, m.template_id))) in {"EQUIVALENT", "COMPLEMENTARY"}
                     or content_key(selected.standard_answer) == content_key(known[m.template_id].standard_answer)) for m in full):
                return result("NEED_CLARIFICATION", "请补充具体适用情况，以便确认准确的标准答复。", "CLARIFICATION", reason="DAY1_MATCH_REQUIRES_CONFIRMATION_NO_QA_OVERRIDE")
            # Same typed subject + scalar dimension must also be checked outside
            # the model's FULL list. A generic Q&A rate can contradict an
            # estate-specific rate even if its standard question is worded
            # differently. Already gated applicable records are compared only.
            scalar_conflicts = scalar_disagreements(selected, templates, scope)
            if scalar_conflicts:
                audit["template_conflicts"] = [t.model_dump() for t in [selected, *scalar_conflicts]]
                return result("CONFLICT", "这项资料有不同答复，目前无法确认，请联系物业核实。", "CLARIFICATION", reason="TYPED_CROSS_SOURCE_SCALAR_CONFLICT")
            related = []
            for m in full:
                other = known[m.template_id]
                if other.template_id == selected.template_id or other.source_type == selected.source_type:
                    continue
                relation = comparisons.get(frozenset((selected.template_id, other.template_id)))
                same_question = canonical_question(selected.canonical_question) == canonical_question(other.canonical_question)
                same_answer = content_key(selected.standard_answer) == content_key(other.standard_answer)
                # Explicit different numerical values for the same fact cannot be
                # hidden by source priority or a model's 'equivalent' label.
                patterns = (r"\d{1,2}[:：]\d{2}\s*[—–至-]\s*\d{1,2}[:：]\d{2}", r"\d+(?:\.\d+)?\s*元", r"0\d{2,3}-\d{7,8}")
                numerical_conflict = (same_question or relation == "EQUIVALENT") and any(
                    (a := set(re.findall(pattern, selected.standard_answer))) and
                    (b := set(re.findall(pattern, other.standard_answer))) and a != b for pattern in patterns)
                if relation == "CONFLICT" or numerical_conflict:
                    audit["template_conflicts"] = [selected.model_dump(), other.model_dump()]
                    return result("CONFLICT", "这项资料有不同答复，目前无法确认，请联系物业核实。", "CLARIFICATION", reason="CROSS_SOURCE_FACT_CONFLICT")
                if same_answer or relation in {"EQUIVALENT", "COMPLEMENTARY"}:
                    related.append(other.template_id)
                    audit.setdefault("coverage_relations", []).append({"day1": selected.template_id, "additional_qa": other.template_id,
                        "relation": relation or "IDENTICAL_ANSWER", "selected_source": selected.source_type,
                        "equivalence_group_id": selected.equivalence_group_id})
            peers = [t for t in templates if canonical_question(t.canonical_question) == canonical_question(selected.canonical_question)
                     and t.conditions == selected.conditions and t.usage_window == selected.usage_window
                     and t.template_id not in related]
            if len({content_key(t.standard_answer) for t in peers}) > 1:
                audit["template_conflicts"] = [t.model_dump() for t in peers]
                return result("CONFLICT", "这项资料有不同答复，目前无法确认，请联系物业核实。", "CLARIFICATION", reason="CONFLICTING_APPLICABLE_STANDARDS")
            rivals = [m for m in full if m.template_id != top.template_id
                and content_key(known[m.template_id].standard_answer) != content_key(selected.standard_answer)
                and m.template_id not in related
                and not set(known[m.template_id].source_ids) <= set(selected.source_ids)]
            margin = top.score - max(m.score for m in rivals) if rivals else 1.0
            audit["template_top"] = {"id": top.template_id, "score": top.score, "margin": margin, "eligible": True}
            threshold_ok = bool(exact) or calibrated and top.score >= threshold and margin >= required_margin
            if rivals and (margin <= 0 or not calibrated or margin < required_margin):
                return result("NEED_CLARIFICATION", "有几项答复可能相关，请补充具体服务和适用情况。", "CLARIFICATION", reason="AMBIGUOUS_TEMPLATE_MATCH")
            if threshold_ok:
                fresh = [c for c in recheck() if in_estate(c)]
                live, _ = project_templates(fresh)
                current = next((t for t in live if t.template_id == selected.template_id and t.version_hash == selected.version_hash), None)
                if not current:
                    return result("NOT_VERIFIED", "这项资料的版本或适用状态发生了变化，需要重新核实。", "CLARIFICATION", reason="TEMPLATE_CHANGED_DURING_MATCH")
                fresh_peers = [t for t in live if canonical_question(t.canonical_question) == canonical_question(current.canonical_question)
                               and t.conditions == current.conditions and t.usage_window == current.usage_window
                               and not (t.template_id in related and t.version_hash == known[t.template_id].version_hash)]
                if len({content_key(t.standard_answer) for t in fresh_peers}) > 1:
                    audit["template_conflicts"] = [t.model_dump() for t in fresh_peers]
                    return result("CONFLICT", "这项资料有不同答复，目前无法确认，请联系物业核实。", "CLARIFICATION", reason="CONFLICT_APPEARED_DURING_MATCH")
                return result("ANSWERED", current.standard_answer, "TEMPLATE", chosen=current, score=top.score, reason=top.reason)
            return result("NEED_CLARIFICATION", "有几项标准答复可能相关，请补充具体服务和适用情况。", "CLARIFICATION", reason="TEMPLATE_AMBIGUOUS_NO_REWRITE")
        facts = [rows[i] for i in dict.fromkeys(decision.knowledge_ids) if usable_text(rows[i])[0] and not rows[i].audit.get("ambiguous")] if decision.sufficiency != "NONE" else []
        relevant = []
        for c in facts:
            aliases = [t for t in templates if c.identity in t.source_ids]
            if any(scope.score(t.canonical_question, t.canonical_question, t.service_code) >= .6 for t in aliases):
                relevant.append(c)
                continue
            qsubjects, csubjects = scope.subjects & SPECIFIC, c.subjects & SPECIFIC
            mismatch = (qsubjects and csubjects and not qsubjects.intersection(csubjects)
                or scope.facets and c.facets and not any(facet_matches(f, c.facets) for f in scope.facets))
            if mismatch:
                audit.setdefault("irrelevant_generation_facts_withheld", []).append(c.summary())
            else:
                relevant.append(c)
        facts = relevant
        for t in templates:
            if not set(t.source_ids).intersection(c.identity for c in facts):
                continue
            scalar_conflicts = scalar_disagreements(t, templates, scope)
            if scalar_conflicts:
                audit["template_conflicts"] = [x.model_dump() for x in [t, *scalar_conflicts]]
                return result("CONFLICT", "这项资料有不同答复，目前无法确认，请联系物业核实。", "CLARIFICATION", reason="GENERATION_SOURCE_SCALAR_CONFLICT")
        # Partial knowledge must not bypass a conflict that would block a full template.
        conflicts = []
        for chosen in facts:
            peers = [c for c in candidates if c.scope == chosen.scope and c.window == chosen.window and
                     (canonical_question(c.question) == canonical_question(chosen.question) or
                      c.origin == chosen.origin == "POLICY" and c.fact.get("fact_key") and c.fact.get("fact_key") == chosen.fact.get("fact_key"))]
            if len({content_key(c.text) for c in peers}) > 1:
                conflicts.extend(c.summary() for c in peers)
        if conflicts:
            audit["template_conflicts"] = unique(conflicts)
            return result("CONFLICT", "这项资料有不同答复，目前无法确认，请联系物业核实。", "CLARIFICATION", reason="CONFLICTING_RELEVANT_KNOWLEDGE")
        if facts:
            fresh = {c.identity: c for c in recheck() if in_estate(c)}
            facts = [fresh[c.identity] for c in facts if c.identity in fresh and fresh[c.identity].version == c.version and fresh[c.identity].text == c.text]
        if facts:
            audit["generation_facts"] = [usable_text(c)[0] for c in facts]
            audit["generation_sufficiency"] = decision.sufficiency
            return result("ANSWERED", "\n".join(audit["generation_facts"]), "KNOWLEDGE_GENERATION", facts=facts,
                          reason="VERIFIED_KNOWLEDGE_WITHOUT_FULL_TEMPLATE")
        if baseline["status"] == "CONFLICT":
            audit["template_conflicts"] = audit.get("conflicts", [])
            return result("CONFLICT", baseline["answer_text"], "CLARIFICATION", reason="UNRESOLVED_RELEVANT_FACT_CONFLICT")
        if baseline["status"] in {"NEED_CLARIFICATION", "NOT_VERIFIED"} and not getattr(self.matcher, "semantic_available", False):
            return result(baseline["status"], baseline["answer_text"], "CLARIFICATION", reason="SOURCE_OR_CONTEXT_NOT_VERIFIED")
        return result("NOT_FOUND", "目前没有足够可靠的信息确认这一点。", "GENERAL_CONVERSATION", reason="NO_TEMPLATE_OR_RELEVANT_EVIDENCE")
