"""Global deduplication, conservative conflict resolution and one source-bound final answer."""

import copy
import re

from ..domain import canonical, digest
from .contract import AnswerResult
from .intent import PARTS, canonical_question, content_key, facet_matches, normalized_answer


def unique(items):
    return list({canonical(item): item for item in items}.values())


def business_conditions(description):
    """Keep verbatim business clauses; extraction/review commentary stays in candidate audit only."""
    clauses = re.split(r"[；;。\n]", description)
    internal = re.compile(r"Policy|Mock|POC|核验|待确认|不推断|不补|不扩大|本条|本轮|原件|资料|不是.*生效|不泛化|仅记录|不作为|保留|不扩展")
    business = re.compile(r"起计算|之日起|起算|合同另有约定|仅适用|不包括|必须|须|无噪音|暂定|预计|限于")
    return [c.strip() for c in clauses if c.strip() and not internal.search(c) and business.search(c)]


def usable_text(candidate):
    text = normalized_answer(candidate.text)
    notes = []
    # A consultation must not claim that an unavailable action was performed, or expose nonexistent UI links.
    promise = re.compile(r"咨询工单|自动回电|稍后.*(?:联系|回电)|专人.*联系|填写下方|点击下方|已安排|已为您|为您登记|为您提交")
    sentences = re.split(r"(?<=[。！？!?])", text)
    kept = [s for s in sentences if s.strip() and not promise.search(s)]
    if len(kept) < len([s for s in sentences if s.strip()]):
        notes.append("UNIMPLEMENTED_ACTION_SENTENCES_WITHHELD")
    text = "".join(kept).strip()
    if re.search(r"_{2,}|[□■]{2,}|\b(?:XXX|TBD)\b|待补充|待填写", text, re.I):
        return "", [*notes, "INCOMPLETE_SOURCE_TEXT"]
    for condition in candidate.public_conditions:
        if content_key(condition) not in content_key(text):
            text += ("" if text.endswith(("。", "；")) else "。") + condition + "。"
    return text, notes


def deduplicate(candidates):
    groups, removed, conflicts = {}, [], []
    identities = {}
    for original in candidates:
        c = copy.deepcopy(original)
        identity = (c.estate_id, c.identity)
        if identity in identities:
            prior = identities[identity]
            if prior.version != c.version or content_key(prior.text) != content_key(c.text):
                conflicts.append({"reason": "ID_CONTENT_MISMATCH", "ids": [prior.identity, c.identity]})
            prior.sources = unique(prior.sources + c.sources)
            removed.append({"id": c.identity, "reason": "SAME_ID"})
            continue
        identities[identity] = c
        key = (c.estate_id, c.origin, c.version, content_key(c.text), canonical(c.scope), c.window,
               canonical(c.public_conditions))
        if key in groups:
            prior = groups[key]
            prior.sources = unique(prior.sources + c.sources)
            prior.audit.setdefault("equivalent_ids", []).append(c.identity)
            prior.audit.setdefault("equivalent_versions", []).append(c.version)
            removed.append({"id": c.identity, "kept": prior.identity, "reason": "SAME_VERSION_CONTENT_SCOPE_WINDOW"})
        else:
            groups[key] = c
    return list(groups.values()), removed, conflicts


def compose(scope, candidates, *, rejected=None, missing=None, warnings=None, context=None):
    rejected, missing = rejected or [], missing or []
    warnings = list(warnings or [])
    context = context or {}
    audit_candidates = [c.summary() for c in candidates]
    candidates, duplicates, conflicts = deduplicate(candidates)
    selected = []

    def finish(status, text, extra=None):
        ids = unique([i for c in selected for i in [c.identity, *c.audit.get("equivalent_ids", [])] if ":POL:" in i])
        refs = unique([s for c in selected for s in c.sources])
        debug = {"question": scope.question, "service_code": scope.service_code or scope.inferred_service_code,
                 "explicit_service_code": scope.service_code,
                 "subjects": sorted(scope.subjects), "requested_facets": sorted(scope.facets), "context": context,
                 "candidates": audit_candidates, "rejected": rejected, "missing_context": missing,
                 "duplicates": duplicates, "conflicts": conflicts, "selected_ids": [c.identity for c in selected],
                 "matched_faq_ids": [c.identity for c in selected if ":FAQ:" in c.identity],
                 "raw_matches": [{**c.match, "sources": c.sources} for c in selected], **(extra or {})}
        retained_warnings = warnings + [note for c in selected for note in c.notes + c.audit.get("blocking_issues", [])]
        return AnswerResult(status=status, answer_text=text, estate_id=scope.estate_id,
                            matched_policy_ids=ids, evidence_refs=refs, warnings=list(dict.fromkeys(retained_warnings)),
                            debug_info=debug)

    if scope.estate_id is None:
        return finish("NEED_CLARIFICATION", "请问您咨询的是烟台港城华府，还是青岛国际健康生态谷？")
    if context.get("scope_mismatch"):
        return finish("NEED_CLARIFICATION", "您提到的楼盘与当前登录楼盘不同，请先切换到对应楼盘登录。")
    if context.get("housing_mismatch"):
        return finish("NEED_CLARIFICATION", "问题中的房屋类型与当前选择的房屋不一致，请先确认房屋类型或切换房屋。")
    required_policy_context = [m for m in missing if ":POL:" in m.get("id", "")]
    if required_policy_context:
        fields = {m["field"] for m in required_policy_context}
        text = ("请先选择装修活动类型，例如毛坯装修、简单装修、拆改或地暖施工。" if "renovation_activity" in fields
                else "请先选择您的房屋，以确认房屋类型、电梯配置及适用标准。")
        return finish("NEED_CLARIFICATION", text)
    candidates = [c for c in candidates if c.estate_id == scope.estate_id and c.score >= 0.6]
    if "warranty" in scope.subjects and not scope.subjects.intersection(PARTS) and len(candidates) > 1:
        return finish("NEED_CLARIFICATION", "请问您想了解哪个部位的保修期限，例如屋面防水、门窗或供热系统？")
    if conflicts:
        warnings.append("UNRESOLVED_CONFLICT")
        return finish("CONFLICT", "相关答复存在不一致，目前无法确认，请联系物业核实。")
    if not candidates:
        if missing:
            fields = {x["field"] for x in missing}
            text = ("请先选择装修活动类型，例如毛坯装修、简单装修、拆改或地暖施工。" if "renovation_activity" in fields
                    else "请先选择您的房屋，以确认房屋类型、电梯配置及适用标准。")
            return finish("NEED_CLARIFICATION", text)
        if rejected:
            return finish("NOT_VERIFIED", "这项信息暂时无法确认，请联系物业核实。")
        return finish("NOT_FOUND", "暂未找到能回答这个问题的可靠信息，建议联系物业核实。")
    if not scope.facets and len({(c.fact.get("fact_key"), canonical_question(c.question)) for c in candidates}) > 1:
        exact = [c for c in candidates if c.score == 1.0]
        if not exact:
            return finish("NEED_CLARIFICATION", "请补充您想了解的具体事项，例如费用、办理流程、位置或服务时间。")
    wanted = scope.facets or frozenset({"general"})
    uncovered = []
    for facet in sorted(wanted):
        pool = [c for c in candidates if facet == "general" or facet_matches(facet, c.facets) or c.score == 1.0 and c.origin == "FAQ"]
        policies = [c for c in pool if c.origin == "POLICY"]
        direct = [c for c in policies if c.score == 1.0]
        if direct:
            keys = {c.fact.get("fact_key") for c in direct}
            titles = {canonical_question(c.question) for c in direct}
            policies = [c for c in policies if c in direct or c.fact.get("fact_key") in keys
                        or canonical_question(c.question) in titles]
        elif facet == "channels" and any(c.fact.get("predicate") == "channel" for c in policies):
            policies = [c for c in policies if c.fact.get("predicate") == "channel"]
        if facet == "price" and "components" not in wanted:
            policies = [c for c in policies if c.fact.get("value_role") != "COMPONENT"]
        if policies:
            selected.extend(policies)
        elif pool:
            best = max(c.score for c in pool)
            near = [c for c in pool if c.score >= best - 0.02]
            questions = {canonical_question(c.question) for c in near}
            if len(questions) > 1:
                return finish("NEED_CLARIFICATION", "请补充您想了解的具体事项和适用条件，以便确认准确答复。")
            selected.extend(near)
        else:
            uncovered.append(facet)
    selected = list({c.identity: c for c in selected}.values())
    if uncovered or not selected:
        return finish("NOT_VERIFIED", "这个问题涉及的信息尚不完整，暂时无法确认，请联系物业核实。")
    fact_groups = {}
    for c in selected:
        group = (c.estate_id, c.fact.get("fact_key") if c.origin == "POLICY" else canonical_question(c.question))
        fact_groups.setdefault(group, []).append(c)
    for group, rows in fact_groups.items():
        values = {canonical({"value": c.fact.get("value"), "unit": c.fact.get("unit")}) if c.origin == "POLICY"
                  else content_key(c.text) for c in rows}
        if len(values) > 1:
            conflicts.append({"group": group, "reason": "DIFFERENT_APPLICABLE_VALUES", "ids": [c.identity for c in rows],
                              "values": [c.text for c in rows], "scopes": [c.scope for c in rows]})
    question_groups = {}
    for c in selected:
        if c.origin == "POLICY":
            question_groups.setdefault(canonical_question(c.question), []).append(c)
    for question, rows in question_groups.items():
        if len({canonical(c.fact.get("value")) for c in rows}) > 1:
            conflicts.append({"group": question, "reason": "SAME_STANDARD_QUESTION_DIFFERENT_VALUES",
                              "ids": [c.identity for c in rows], "values": [c.text for c in rows]})
    if conflicts:
        warnings.append("UNRESOLVED_CONFLICT")
        return finish("CONFLICT", "相关答复存在不一致，目前无法确认，请联系物业核实。")
    texts = []
    for c in selected:
        if c.audit.get("ambiguous"):
            warnings.append("AMBIGUOUS_SOURCE")
            return finish("NOT_VERIFIED", "这项信息存在未明确的时间或适用条件，请联系物业确认。")
        text, notes = usable_text(c)
        warnings.extend(notes)
        if not text:
            return finish("NOT_VERIFIED", "这项信息暂时无法确认，请联系物业核实。")
        if content_key(text) not in {content_key(t) for t in texts}:
            texts.append(text)
    # Shared-prefix lists are complementary facts, never multiple competing candidate answers.
    if len(texts) > 1 and wanted == {"scope"}:
        prefix = texts[0].split("包括", 1)[0] + "包括" if "包括" in texts[0] else ""
        if prefix and all(t.startswith(prefix) for t in texts):
            texts = [prefix + "；".join(t[len(prefix):].rstrip("。") for t in texts) + "。"]
    elif len(texts) > 1 and wanted == {"channels"}:
        frames = [re.fullmatch(r"(可通过)(.+)(进行[^。]+。)", t) for t in texts]
        if all(frames) and len({(m[1], m[3]) for m in frames}) == 1:
            texts = [frames[0][1] + "、".join(m[2] for m in frames) + frames[0][3]]
    return finish("ANSWERED", "\n".join(texts), {"answer_hash": digest(texts)})
