"""Shared Day1/Day2, versioned source choices authorized by the user's explicit rule."""

import json
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from .domain import ReviewDecisionRecord, canonical, digest, scoped
from .faq_contract import Exact
from .faq_importer import load_faq_packet
from .importer import load_packet, properties

PACKET = "data/day1/source_priority_20261029/selections.json"
RULE = "根据这个规则核验：在day1资料对照里，有day1资料的保留day1的回答，未配置明确的Day1对应条目保留新增Q&A的回答"


class PolicyVersion(Exact):
    policy_id: str
    version_hash: str


class AnswerSelection(Exact):
    selection_version: Literal["day1-priority-v1"]
    selection_id: str
    selection_hash: str
    estate_id: Literal["E001", "E002"]
    faq_id: str
    faq_version_hash: str
    question: str
    selected_source: Literal["DAY1_POLICY", "SUPPLEMENT_QA"]
    policy_versions: list[PolicyVersion]
    selected_answer: str
    selected_conditions: str
    source_references: list[dict]
    review_notes: str
    has_distinct_answers: bool
    human_rule: Literal["根据这个规则核验：在day1资料对照里，有day1资料的保留day1的回答，未配置明确的Day1对应条目保留新增Q&A的回答"]
    review_purpose: Literal["POC_DEMO"]

    @model_validator(mode="after")
    def version_and_scope(self):
        payload = self.model_dump(exclude={"selection_id", "selection_hash"})
        if self.selection_hash != digest(payload) or self.selection_id != f"{self.estate_id}:SELECTION:{self.selection_hash}":
            raise ValueError("来源选择版本失效")
        if not scoped(self.faq_id, self.estate_id) or any(not scoped(p.policy_id, self.estate_id) for p in self.policy_versions):
            raise ValueError("来源选择跨楼盘")
        if bool(self.policy_versions) != (self.selected_source == "DAY1_POLICY"):
            raise ValueError("Day1优先选择缺少政策版本或错误混用Q&A")
        if not self.selected_answer or not self.source_references:
            raise ValueError("来源选择缺少答复或来源")
        return self


def make_selection(c, policies, *, notes, has_distinct_answers):
    selected = [policies[key] for key in c["baseline_comparison_ids"]]
    if any(p["estate_id"] != c["estate_id"] for p in selected):
        raise ValueError("不能跨楼盘借用Day1")
    if selected:
        answer = "\n\n".join((f'{i}. {p["title"]}\n' if len(selected) > 1 else "") + p["statement"] for i, p in enumerate(selected, 1))
        conditions = "\n\n".join(p["title"] + "：" + p["conditions"]["description"] for p in selected)
        sources = [{"source_file": p["source_file"], "source_locator": p["source_locator"],
                    "policy_id": p["policy_id"], "version_hash": p["version_hash"]} for p in selected]
    else:
        answer = c["answer_candidate"]
        conditions = "保留新增Q&A原答复中的适用条件和限定，不补写原文缺失内容。"
        sources = [{"source_file": c["source_file"], "source_locator": c["source_locator"],
                    "document_id": c["document_id"], "chunk_id": c["chunk_id"], "source_sha256": c["source_sha256"]}]
    r = {"selection_version": "day1-priority-v1", "estate_id": c["estate_id"], "faq_id": c["faq_id"],
         "faq_version_hash": c["version_hash"], "question": c["question"],
         "selected_source": "DAY1_POLICY" if selected else "SUPPLEMENT_QA",
         "policy_versions": [{"policy_id": p["policy_id"], "version_hash": p["version_hash"]} for p in selected],
         "selected_answer": answer, "selected_conditions": conditions, "source_references": sources,
         "review_notes": notes, "has_distinct_answers": has_distinct_answers,
         "human_rule": RULE, "review_purpose": "POC_DEMO"}
    r["selection_hash"] = digest(r)
    r["selection_id"] = f'{c["estate_id"]}:SELECTION:{r["selection_hash"]}'
    return AnswerSelection.model_validate(r).model_dump()


def load_selections(root):
    root = Path(root)
    packet = json.loads((root / PACKET).read_text())
    candidates, _ = load_faq_packet(root)
    known = {c["faq_id"]: c for c in candidates}
    policies = {p["policy_id"]: p for p in load_packet(root)["policies"]}
    result = []
    for raw in packet["selections"]:
        r = AnswerSelection.model_validate(raw).model_dump()
        c = known.get(r["faq_id"])
        if not c or c["estate_id"] != r["estate_id"] or c["version_hash"] != r["faq_version_hash"]:
            raise ValueError("来源选择的FAQ版本已变化")
        expected = make_selection(c, policies, notes=r["review_notes"], has_distinct_answers=r["has_distinct_answers"])
        if r != expected:
            raise ValueError("来源选择未遵循已配置的Day1优先规则或原文发生变化")
        result.append(r)
    if not result or len({r["faq_id"] for r in result}) != len(result):
        raise ValueError("来源选择为空或重复")
    return result


def build_selection_receipt(root, *, recorded_at):
    root = Path(root)
    rows = load_selections(root)
    r = {"receipt_version": "explicit-human-source-priority-v1", "human_origin": "USER_EXPLICIT_CONFIRMATION",
         "human_statement": RULE, "sync_instruction": "先把刚刚修改过的文件同步到day1和day2中，并且再对齐Ontology Schema（本体结构设计）和知识抽取及审核表",
         "review_purpose": "POC_DEMO", "recorded_at": recorded_at, "demo_valid_from": "2026-10-29",
         "demo_valid_to": None, "real_policy_effective_dates_confirmed": False,
         "packet_file": PACKET, "packet_sha256": digest((root / PACKET).read_bytes()),
         "targets": [{"estate_id": x["estate_id"], "faq_id": x["faq_id"], "selection_hash": x["selection_hash"]} for x in rows]}
    r["batch_id"] = "source-priority-" + digest(r)[:24]
    r["receipt_hash"] = digest(r)
    return r


def validate_selection_receipt(root, receipt, ack):
    if not ack or ack != receipt.get("receipt_hash") or digest({k: v for k, v in receipt.items() if k != "receipt_hash"}) != ack:
        raise ValueError("必须绑定用户明确给出的来源规则与同步授权")
    if (receipt.get("receipt_version") != "explicit-human-source-priority-v1"
        or receipt.get("human_origin") != "USER_EXPLICIT_CONFIRMATION" or receipt.get("human_statement") != RULE
        or receipt.get("review_purpose") != "POC_DEMO" or receipt.get("real_policy_effective_dates_confirmed") is not False
        or not receipt.get("sync_instruction")):
        raise ValueError("来源选择只能按用户规则用于POC")
    if datetime.fromisoformat(receipt["recorded_at"]).tzinfo is None:
        raise ValueError("来源审核时间必须包含时区")
    start = date.fromisoformat(receipt["demo_valid_from"])
    if receipt["demo_valid_to"] and date.fromisoformat(receipt["demo_valid_to"]) < start:
        raise ValueError("来源审核演示区间无效")
    if receipt["packet_file"] != PACKET or digest((Path(root) / PACKET).read_bytes()) != receipt["packet_sha256"]:
        raise ValueError("已核验的来源选择包已改变")
    rows = load_selections(root)
    targets = [{"estate_id": r["estate_id"], "faq_id": r["faq_id"], "selection_hash": r["selection_hash"]} for r in rows]
    if receipt["targets"] != targets:
        raise ValueError("来源审核目标不能新增、删除、改版本或跨楼盘")
    return rows


def publish_selections(repo, auth, root, receipt, *, human_ack_hash):
    rows = validate_selection_receipt(root, receipt, human_ack_hash)
    candidates, _ = load_faq_packet(root)
    known = {c["faq_id"]: dict(c, runtime_imported=True) for c in candidates}
    scopes = sorted({r["estate_id"] for r in rows})
    with auth.connect() as db:
        db.execute("CREATE TABLE IF NOT EXISTS human_review_receipts (estate_id TEXT NOT NULL,batch_id TEXT NOT NULL,receipt_hash TEXT NOT NULL,receipt_json TEXT NOT NULL,state TEXT NOT NULL,PRIMARY KEY(estate_id,batch_id))")
        for scope in scopes:
            old = db.execute("SELECT receipt_hash FROM human_review_receipts WHERE estate_id=? AND batch_id=?", (scope, receipt["batch_id"])).fetchone()
            if old and old["receipt_hash"] != receipt["receipt_hash"]:
                raise ValueError("同一批次不能对应不同来源审核")
            db.execute("INSERT OR IGNORE INTO human_review_receipts VALUES(?,?,?,?,?)", (scope, receipt["batch_id"], receipt["receipt_hash"], canonical(receipt), "PREPARED"))
            db.execute("INSERT OR IGNORE INTO review_grants VALUES(?,?,?,1)", (scope, f"{scope}:REVIEWER:PROJECT_USER", f"{scope}:AUTHZ:{receipt['batch_id']}"))
            if not db.execute("SELECT active FROM review_grants WHERE estate_id=? AND actor_ref=? AND authorization_ref=?", (scope, f"{scope}:REVIEWER:PROJECT_USER", f"{scope}:AUTHZ:{receipt['batch_id']}")).fetchone()["active"]:
                raise ValueError("来源审核权限已撤销")

    def transaction(tx):
        for r in rows:
            scope, fid = r["estate_id"], r["faq_id"]
            live = tx.run("MATCH (f:FAQ {estate_id:$estate_id,faq_id:$id}) SET f._selection_lock=coalesce(f._selection_lock,0)+1 RETURN properties(f) AS f", estate_id=scope, id=fid).single()
            if not live or any(live["f"].get(k) != v for k, v in properties(known[fid], True).items()):
                raise ValueError("FAQ内容发生变化，不能沿用来源审核")
            for p in r["policy_versions"]:
                existing = tx.run("MATCH (p:Policy {estate_id:$estate_id,policy_id:$pid,version_hash:$hash}) RETURN count(p) AS n", estate_id=scope, pid=p["policy_id"], hash=p["version_hash"]).single()["n"]
                if existing != 1:
                    raise ValueError("Day1目标缺失、跨楼盘或版本变化")
            review = {"estate_id": scope, "review_decision_id": f'{scope}:REVIEW:{digest([receipt["batch_id"], r["selection_id"]])}',
                      "actor_ref": f"{scope}:REVIEWER:PROJECT_USER", "actor_role": "HUMAN_REVIEWER",
                      "authorization_ref": f"{scope}:AUTHZ:{receipt['batch_id']}", "target_type": "FAQ",
                      "target_component": "ANSWER_SELECTION", "target_id": fid, "target_version_hash": r["selection_hash"],
                      "decision": "APPROVED", "note": RULE + "；回执SHA256=" + receipt["receipt_hash"],
                      "decided_at": receipt["recorded_at"], "is_mock": False, "review_purpose": "POC_DEMO", "review_batch_id": receipt["batch_id"]}
            review["version_hash"] = digest({k: v for k, v in review.items() if k != "review_decision_id"})
            ReviewDecisionRecord.model_validate(review)
            stored = tx.run("MERGE (r:ReviewDecision {estate_id:$estate_id,review_decision_id:$rid}) ON CREATE SET r += $props RETURN properties(r) AS r", estate_id=scope, rid=review["review_decision_id"], props=review).single()["r"]
            if stored != review:
                raise ValueError("来源审核事件与已有记录冲突")
            tx.run("MATCH (f:FAQ {estate_id:$estate_id,faq_id:$id}),(r:ReviewDecision {estate_id:$estate_id,review_decision_id:$rid}) MERGE (r)-[:REVIEWS {estate_id:$estate_id}]->(f) SET f.answer_selection_json=$selection,f.answer_selection_hash=$hash,f.answer_selection_source=$source,f.answer_selection_batch_id=$batch,f.answer_selection_review_decision_id=$rid,f.answer_selection_valid_from=$start,f.answer_selection_valid_to=$end", estate_id=scope, id=fid, rid=review["review_decision_id"], selection=canonical(r), hash=r["selection_hash"], source=r["selected_source"], batch=receipt["batch_id"], start=receipt["demo_valid_from"], end=receipt["demo_valid_to"]).consume()
            for p in r["policy_versions"]:
                tx.run("MATCH (f:FAQ {estate_id:$estate_id,faq_id:$id}),(p:Policy {estate_id:$estate_id,policy_id:$pid,version_hash:$hash}) MERGE (f)-[r:ANSWERED_BY {estate_id:$estate_id,selection_hash:$selection_hash}]->(p)", estate_id=scope, id=fid, pid=p["policy_id"], hash=p["version_hash"], selection_hash=r["selection_hash"]).consume()
        return len(rows)
    with repo.driver.session(database=repo.settings.database) as session:
        count = session.execute_write(transaction)
    with auth.connect() as db:
        for scope in scopes:
            db.execute("UPDATE human_review_receipts SET state='PUBLISHED' WHERE estate_id=? AND batch_id=?", (scope, receipt["batch_id"]))
    return {"batch_id": receipt["batch_id"], "selection_records": count,
            "source_counts": dict(Counter(r["selected_source"] for r in rows)), "real_policy_dates_changed": False,
            "original_faq_versions_changed": False, "human_receipt_hash": receipt["receipt_hash"]}


def verified_selection(bundle, scope, day, root, reviewer_authorized, policy_bundles, unit=None, building=None, renovation_activity=None):
    """Verify both human source selection and the original FAQ evidence; never trusts a JSON flag alone."""
    from .faq_knowledge import verified_demo_faq_source
    from .knowledge import _review_matches

    try:
        f = bundle["faq"]
        r = AnswerSelection.model_validate_json(f["answer_selection_json"]).model_dump()
        if r["estate_id"] != scope or r["faq_id"] != f["faq_id"] or r["faq_version_hash"] != f["version_hash"]:
            return None
        if r["selection_hash"] != f.get("answer_selection_hash") or r["selected_source"] != f.get("answer_selection_source") or not f.get("answer_selection_batch_id"):
            return None
        if not f.get("answer_selection_valid_from") or date.fromisoformat(f["answer_selection_valid_from"]) > day:
            return None
        if f.get("answer_selection_valid_to") and date.fromisoformat(f["answer_selection_valid_to"]) < day:
            return None
        if not any(review.get("review_decision_id") == f.get("answer_selection_review_decision_id") and _review_matches(
            review, scope, "FAQ", f["faq_id"], r["selection_hash"], "POC_DEMO", day, reviewer_authorized,
            f["answer_selection_batch_id"], component="ANSWER_SELECTION") for review in bundle["reviews"]):
            return None
        c = json.loads(f["_record_json"])
        policies = {p["policy"]["policy_id"]: json.loads(p["policy"]["_record_json"]) for p in policy_bundles}
        expected = make_selection(c, policies, notes=r["review_notes"], has_distinct_answers=r["has_distinct_answers"])
        if expected != r:
            return None
        targets = {(p["policy_id"], p["version_hash"]) for p in r["policy_versions"]}
        edges = {(p["policy_id"], p["version_hash"]) for p in bundle.get("selection_policies", []) if p.get("selection_hash") == r["selection_hash"]}
        if targets != edges:
            return None
        if not verified_demo_faq_source(bundle, scope, day, root, reviewer_authorized, unit, building, renovation_activity):
            return None
        return r
    except (ValueError, KeyError, TypeError):
        return None
