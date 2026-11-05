"""A single lineage-preserving view over eligible Day1 Policy and additional FAQ records."""

import re
from typing import Literal

from pydantic import Field

from ..domain import Strict, canonical, digest
from .intent import canonical_question, facets, normalized_answer
from .selector import business_conditions, deduplicate, unique, usable_text


class Template(Strict):
    template_id: str
    estate_id: str
    service_code: str
    canonical_question: str
    question_variants: list[str] = Field(default_factory=list)
    standard_answer: str
    conditions: dict
    source_type: Literal["DAY1", "ADDITIONAL_QA"]
    source_priority: int
    source_document_id: str
    source_locator: dict
    linked_policy_ids: list[str]
    review_status: Literal["APPROVED", "APPROVED_FOR_DEMO"]
    usage_authorization: dict
    version_hash: str
    valid_from: str | None
    valid_to: str | None
    conflict_status: str = "NOT_DETECTED"
    equivalence_group_id: str
    topic: str
    priority: int
    usage_mode: str
    usage_window: tuple[str | None, str | None]
    evidence_refs: list[dict]
    source_ids: list[str]
    fact_keys: list[str] = Field(default_factory=list)
    answer_fragments: list[dict] = Field(default_factory=list)
    human_source_binding: dict | None = None


def normalized_conditions(scope):
    fields = ("estate_id", "residence_category", "has_elevator", "building_form", "renovation_activity", "warranty_item")
    return {k: None if scope.get(k) == "UNSPECIFIED" else scope.get(k) for k in fields}


def binding_coverage(selection, rows, issues=()):
    """A human source choice is lineage, not proof every clause was answered.

    Only certify an alias when its requested fact dimensions are represented by
    the selected records. Uncertain compound mappings remain available as facts.
    """
    requested = set(facets(selection["question"]))
    offered = set().union(*(facets(c.question + " " + c.text, fact=c.fact) for c in rows))
    missing = requested - offered
    if re.search(r"找谁|谁(?:来|负责)|由谁", selection["question"]) and not any(
            c.fact.get("predicate") == "responsibility" for c in rows):
        missing.add("responsibility")
    clauses = [v for v in re.split(r"[？?；;，,]", selection["question"]) if v.strip()]
    if "PARTIAL_OVERLAP_WITH_BASELINE" in issues and len(clauses) > 1:
        missing.add("compound_coverage_not_certified")
    return {"complete": not missing, "missing_dimensions": sorted(missing),
            "requested_dimensions": sorted(requested), "covered_dimensions": sorted(offered)}


def project_templates(candidates):
    candidates, _, conflicts = deduplicate(candidates)
    if conflicts:
        return [], [{"reason": "ID_CONTENT_CONFLICT", "conflicts": conflicts}]
    groups, excluded = {}, []
    by_id = {c.identity: c for c in candidates}
    for c in candidates:
        text, notes = usable_text(c)
        structural_issues = {"TRUNCATED_OR_PLACEHOLDER_CONTENT", "QUESTION_CONTEXT_MISSING", "MULTIPLE_TOPICS_IN_ONE_CELL"}.intersection(c.audit.get("blocking_issues", []))
        if c.origin == "FAQ" and structural_issues:
            excluded.append({"id": c.identity, "reason": "STANDARD_PAIR_INCOMPLETE", "issues": sorted(structural_issues)})
            continue
        if not text or notes or c.audit.get("ambiguous"):
            excluded.append({"id": c.identity, "reason": "NOT_A_VERBATIM_SAFE_STANDARD", "notes": notes})
            continue
        # Required clauses of the SAME reviewed Policy belong to its approved
        # record. Never mix in another source or let a model rewrite those clauses.
        if c.origin != "POLICY" and normalized_answer(c.text) != text:
            excluded.append({"id": c.identity, "reason": "STANDARD_REQUIRES_UNREVIEWED_REWRITE"})
            continue
        if c.origin == "POLICY" and normalized_answer(c.text) != text and set(c.public_conditions) != set(
            business_conditions(c.audit.get("source_record", {}).get("conditions", {}).get("description", ""))):
            excluded.append({"id": c.identity, "reason": "REQUIRED_CONDITION_NOT_IN_REVIEWED_RECORD"})
            continue
        bindings = c.audit.get("source_bindings") or ([c.audit["answer_selection"]] if c.audit.get("answer_selection") else [])
        complete = []
        for s in bindings:
            selected_rows = [by_id[p["policy_id"]] for p in s["policy_versions"] if p["policy_id"] in by_id]
            coverage = binding_coverage(s, selected_rows, c.audit.get("binding_issues", {}).get(s["selection_hash"], []))
            if len(selected_rows) == len(s["policy_versions"]) and coverage["complete"]:
                complete.append(s)
            else:
                excluded.append({"id": c.identity, "selection_hash": s["selection_hash"],
                    "question": s["question"], "reason": "SOURCE_BINDING_PARTIAL_COVERAGE", "coverage": coverage})
        bindings = complete
        singles = [s for s in bindings if s["selected_source"] == "DAY1_POLICY" and len(s["policy_versions"]) == 1]
        selection = min(singles, key=lambda s: s["selection_hash"]) if singles else {}
        day1_binding = selection.get("selected_source") == "DAY1_POLICY"
        target_ids = sorted(p["policy_id"] for p in selection.get("policy_versions", []))
        group = groups.setdefault((c.identity, canonical(c.scope), c.window), {"rows": [], "variants": [], "binding": None})
        group["rows"].append(c)
        if day1_binding and len(target_ids) == 1:
            group["binding"] = selection
            group["variants"].extend([s["question"] for s in singles])
        for selected in bindings:
            ids = sorted(p["policy_id"] for p in selected["policy_versions"])
            if selected["selected_source"] == "DAY1_POLICY" and len(ids) > 1:
                alias = c.estate_id + ":TEMPLATE:" + digest(ids)
                bound = groups.setdefault((alias, "AUTHORIZED_MULTI_RECORD", (None, None)), {"rows": [], "variants": [], "binding": selected})
                bound["rows"].append(c)
    templates = []
    for (identity, _, window), group in groups.items():
        rows = list({c.identity: c for c in group["rows"]}.values())
        rows.sort(key=lambda c: c.identity)
        c, binding = rows[0], group["binding"]
        multi = bool(binding and len(binding["policy_versions"]) > 1)
        if multi and {r.identity for r in rows} != {p["policy_id"] for p in binding["policy_versions"]}:
            excluded.append({"id": identity, "reason": "INCOMPLETE_AUTHORIZED_TEMPLATE"})
            continue
        refs = unique([s for r in rows for s in r.sources])
        if not refs:
            continue
        source = "DAY1" if c.origin == "POLICY" else "ADDITIONAL_QA"
        question = min(set(group["variants"]), key=lambda q: (len(q), q)) if binding and not multi and group["variants"] else binding["question"] if binding else c.question
        variants = sorted(({c.question, *group["variants"]} if not multi else set(group["variants"])) - {question})
        # A multi-record standard is allowed only when the existing immutable
        # HUMAN selection approved this exact full answer. No new concatenation.
        standard = binding["selected_answer"].strip() if multi else usable_text(c)[0]
        conditions = normalized_conditions(c.scope)
        if multi:
            scopes = {r.identity: normalized_conditions(r.scope) for r in rows}
            conditions = {k: v if all(s.get(k) == v for s in scopes.values()) else None for k, v in conditions.items()}
            conditions["per_policy_conditions"] = scopes
            starts = [r.window[0] for r in rows if r.window[0]]
            ends = [r.window[1] for r in rows if r.window[1]]
            window = (max(starts) if starts else None, min(ends) if ends else None)
        fact_keys = sorted({r.fact["fact_key"] for r in rows if r.fact.get("fact_key")})
        mode = c.audit.get("approved_mode", "current")
        authorization = {"mode": mode, "purpose": "POC_DEMO" if mode == "demo" else "CURRENT_POLICY",
            "window": window, "source_versions": {r.identity: r.version for r in rows},
            "review_references": sorted({e.get("review_decision_id") for r in rows for e in r.audit.get("review_events", [])
                if e.get("review_decision_id") and e.get("target_id") == r.identity and e.get("target_version_hash") == r.version
                and e.get("review_purpose", "CURRENT_POLICY") == ("POC_DEMO" if mode == "demo" else "CURRENT_POLICY")}),
            "human_selection_hash": binding.get("selection_hash") if binding else None}
        authorization["original_scope_status"] = c.audit.get("scope_status")
        authorization["original_issue_history"] = sorted(c.audit.get("blocking_issues", []))
        if binding and not multi:
            authorization["human_selection_hashes"] = sorted({s["selection_hash"] for s in c.audit.get("source_bindings", [])
                if len(s["policy_versions"]) == 1})
        version = digest({"ids": authorization["source_versions"], "answer": standard, "conditions": conditions,
                          "authorization": authorization, "question": question, "variants": variants})
        equivalent = fact_keys or [canonical_question(question)]
        group_id = c.estate_id + ":EQ:" + digest({"facts": equivalent, "conditions": conditions, "window": window})
        record = c.audit.get("source_record", {})
        templates.append(Template(template_id=identity, estate_id=c.estate_id, service_code=c.service_code,
            canonical_question=question, question_variants=variants, standard_answer=standard, conditions=conditions,
            source_type=source, source_priority=100 if source == "DAY1" else 50,
            source_document_id=refs[0]["document_id"], source_locator=refs[0]["locator"],
            linked_policy_ids=list(dict.fromkeys(i for r in rows if r.origin == "POLICY" for i in [r.identity, *r.audit.get("equivalent_ids", [])])),
            review_status="APPROVED_FOR_DEMO" if mode == "demo" else "APPROVED", usage_authorization=authorization,
            version_hash=version, valid_from=record.get("valid_from"), valid_to=record.get("valid_to"),
            equivalence_group_id=group_id, topic=record.get("topic", "faq"), priority=110 if binding else 50,
            conflict_status=record.get("conflict_status", "NOT_DETECTED"),
            usage_mode=mode, usage_window=window, evidence_refs=refs, source_ids=[r.identity for r in rows], fact_keys=fact_keys,
            human_source_binding={"selection_hash": binding.get("selection_hash"), "question": binding["question"],
                "authorized_answer": binding["selected_answer"]} if binding else None,
            answer_fragments=[{"record_id": r.identity, "version": r.version, "field": "statement" if r.origin == "POLICY" else "reviewed_answer",
                               "text": r.text, "required_conditions": r.public_conditions} for r in rows]))
    return templates, excluded
