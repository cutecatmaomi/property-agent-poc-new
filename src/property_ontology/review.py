"""Record an explicit HUMAN confirmation for POC-only use. Never invent real effective dates."""

from datetime import date, datetime
from pathlib import Path

from .domain import CORE_IDS, ReviewDecisionRecord, canonical, digest, scoped
from .importer import load_packet


def build_demo_receipt(root, *, statement, usage_confirmation, recorded_at, demo_from, demo_to=None):
    """Author a receipt only; this function does not change a policy or grant approval."""
    root = Path(root)
    packet = load_packet(root)
    report = "outputs/schema_alignment_20261027/reports/knowledge_review.md"
    receipt = {
        "receipt_version": "explicit-human-demo-v1",
        "review_purpose": "POC_DEMO",
        "human_origin": "USER_EXPLICIT_CONFIRMATION",
        "human_statement": statement,
        "human_usage_confirmation": usage_confirmation,
        "reviewer_label": "本项目对话用户（明确人工确认）",
        "recorded_at": recorded_at,
        "demo_valid_from": demo_from,
        "demo_valid_to": demo_to,
        "real_policy_effective_dates_confirmed": False,
        "report_file": report,
        "report_sha256": digest((root / report).read_bytes()),
        "targets": [
            {"estate_id": p["estate_id"], "policy_id": p["policy_id"], "version_hash": p["version_hash"]}
            for p in packet["policies"]
        ],
    }
    receipt["batch_id"] = "demo-" + digest(receipt)[:24]
    receipt["receipt_hash"] = digest(receipt)
    return receipt


def validate_demo_receipt(root, receipt, human_ack_hash):
    if human_ack_hash != receipt.get("receipt_hash") or not human_ack_hash:
        raise ValueError("必须显式确认这份人工审核回执的完整哈希")
    if receipt["receipt_hash"] != digest({k: v for k, v in receipt.items() if k != "receipt_hash"}):
        raise ValueError("人工审核回执被修改")
    if (
        receipt.get("receipt_version") != "explicit-human-demo-v1"
        or receipt.get("review_purpose") != "POC_DEMO"
        or receipt.get("human_origin") != "USER_EXPLICIT_CONFIRMATION"
        or receipt.get("real_policy_effective_dates_confirmed") is not False
        or not receipt.get("human_statement")
        or not receipt.get("human_usage_confirmation")
    ):
        raise ValueError("本入口只处理有明确人工确认的POC演示审核，不能发布真实现行政策")
    if datetime.fromisoformat(receipt["recorded_at"]).tzinfo is None:
        raise ValueError("审核记录时间必须有时区")
    start = date.fromisoformat(receipt["demo_valid_from"])
    if receipt.get("demo_valid_to") and date.fromisoformat(receipt["demo_valid_to"]) < start:
        raise ValueError("演示有效期不合法")
    root = Path(root).resolve()
    report = (root / receipt["report_file"]).resolve()
    expected = root / "outputs/schema_alignment_20261027/reports/knowledge_review.md"
    if report != expected or digest(report.read_bytes()) != receipt["report_sha256"]:
        raise ValueError("知识审核报告与人工确认时的版本不一致")
    packet = load_packet(root)  # Independently replay raw documents BEFORE any authorization/write.
    known = {p["policy_id"]: p for p in packet["policies"]}
    targets = receipt["targets"]
    if not targets or len({t["policy_id"] for t in targets}) != len(targets):
        raise ValueError("人工确认目标为空或重复")
    for target in targets:
        p = known.get(target["policy_id"])
        if not p or p["estate_id"] != target["estate_id"] or p["version_hash"] != target["version_hash"]:
            raise ValueError("人工审核目标版本或楼盘不匹配")
    return packet, [known[t["policy_id"]] for t in targets]


def _decision(receipt, scope, label, target_id, target_hash):
    row = {
        "estate_id": scope,
        "review_decision_id": f"{scope}:REVIEW:{digest([receipt['batch_id'], label, target_id, target_hash])}",
        "actor_ref": f"{scope}:REVIEWER:PROJECT_USER",
        "actor_role": "HUMAN_REVIEWER",
        "authorization_ref": f"{scope}:AUTHZ:{receipt['batch_id']}",
        "target_type": label,
        "target_id": target_id,
        "target_version_hash": target_hash,
        "decision": "APPROVED" if label in {"Policy", "FAQ"} else "VERIFIED",
        "note": "人工确认录入（仅限POC演示）；原文限定保留；回执SHA256=" + receipt["receipt_hash"],
        "decided_at": receipt["recorded_at"],
        "is_mock": False,
        "review_purpose": "POC_DEMO",
        "review_batch_id": receipt["batch_id"],
    }
    row["version_hash"] = digest({k: v for k, v in row.items() if k != "review_decision_id"})
    ReviewDecisionRecord.model_validate(row)
    return row


def publish_demo_review(repo, auth, root, receipt, *, human_ack_hash):
    """Privileged local-admin path, intentionally absent from PropertyActions and future LLM tools."""
    packet, selected = validate_demo_receipt(root, receipt, human_ack_hash)
    chunks = {
        c["chunk_id"]: c
        for c in packet["evidence_chunks"]
        if any(c["chunk_id"] in p["chunk_ids"] for p in selected)
    }
    doc_ids = {c["document_id"] for c in chunks.values()}
    documents = {d["document_id"]: d for d in packet["documents"] if d["document_id"] in doc_ids}
    records = (
        [("Document", d) for d in documents.values()]
        + [("EvidenceChunk", c) for c in chunks.values()]
        + [("Policy", p) for p in selected]
    )
    return publish_demo_records(repo, auth, receipt, records)


def publish_demo_records(repo, auth, receipt, records):
    """Commit prevalidated human decisions atomically; only called by local review publishers."""
    scopes = sorted({r["estate_id"] for _, r in records})
    decisions = {
        (label, r[CORE_IDS[label]]): _decision(
            receipt, r["estate_id"], label, r[CORE_IDS[label]], r["version_hash"]
        )
        for label, r in records
    }

    with auth.connect() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS human_review_receipts (
            estate_id TEXT NOT NULL,batch_id TEXT NOT NULL,receipt_hash TEXT NOT NULL,receipt_json TEXT NOT NULL,
            state TEXT NOT NULL,PRIMARY KEY(estate_id,batch_id))""")
        for scope in scopes:
            old = db.execute(
                "SELECT receipt_hash FROM human_review_receipts WHERE estate_id=? AND batch_id=?",
                (scope, receipt["batch_id"]),
            ).fetchone()
            if old and old["receipt_hash"] != receipt["receipt_hash"]:
                raise ValueError("同一批次对应不同人工意见")
            db.execute(
                "INSERT OR IGNORE INTO human_review_receipts VALUES(?,?,?,?,?)",
                (scope, receipt["batch_id"], receipt["receipt_hash"], canonical(receipt), "PREPARED"),
            )
            db.execute(
                "INSERT OR IGNORE INTO review_grants VALUES(?,?,?,1)",
                (scope, f"{scope}:REVIEWER:PROJECT_USER", f"{scope}:AUTHZ:{receipt['batch_id']}"),
            )
            grant = db.execute(
                "SELECT active FROM review_grants WHERE estate_id=? AND actor_ref=? AND authorization_ref=?",
                (scope, f"{scope}:REVIEWER:PROJECT_USER", f"{scope}:AUTHZ:{receipt['batch_id']}"),
            ).fetchone()
            if not grant["active"]:
                raise ValueError("审核授权已撤销，不能重新启用")

    def transaction(tx):
        existing_decisions = 0
        for label, original in records:
            scope, key = original["estate_id"], CORE_IDS[label]
            identifier = original[key]
            if not scoped(identifier, scope):
                raise ValueError("审核目标跨楼盘")
            live_row = tx.run(
                f"MATCH (n:{label} {{estate_id:$estate_id,{key}:$id}}) "
                "SET n._review_lock=coalesce(n._review_lock,0)+1 RETURN properties(n) AS n",
                estate_id=scope,
                id=identifier,
            ).single()
            if not live_row:
                raise ValueError("审核目标没有导入图谱")
            live = live_row["n"]
            if live.get("version_hash") != original["version_hash"] or live.get("_record_json") != canonical(
                original
            ):
                raise ValueError("图谱内容已变化，旧审核意见不能继承")
            if label == "Policy":
                if (
                    live.get("status") not in {"PENDING_REVIEW", "APPROVED"}
                    or live.get("review_status") != live.get("status")
                    or (live.get("status") == "APPROVED" and live.get("review_purpose") != "POC_DEMO")
                    or live.get("statement") != original["statement"]
                    or live.get("conditions_json") != canonical(original["conditions"])
                ):
                    raise ValueError("政策状态或内容已变更，停止批量审核")
            elif label == "FAQ":
                from .importer import properties
                if live.get("review_status") != "PENDING_REVIEW" or any(
                    live.get(k) != v for k, v in properties(original, True).items()
                ):
                    raise ValueError("FAQ原始内容或正式状态已变化，旧演示审核不能继承")
            elif live.get("verification_status") not in {"PENDING_VERIFICATION", "VERIFIED"}:
                raise ValueError("原件/证据已拒绝或替代，不能套用旧审核")
            if label == "EvidenceChunk" and live.get("extraction_quality") != "NATIVE_TEXT_EXTRACTED":
                raise ValueError("本批不发布扫描或未可靠识别的知识")
            decision = decisions[label, identifier]
            row = tx.run(
                "MERGE (r:ReviewDecision {estate_id:$estate_id,review_decision_id:$rid}) "
                "ON CREATE SET r += $props RETURN properties(r) AS r",
                estate_id=scope,
                rid=decision["review_decision_id"],
                props=decision,
            ).single()["r"]
            if row != decision:
                raise ValueError("已有审核事件与回执不符")
            # Every data match and relationship write independently includes the estate scope.
            tx.run(
                f"MATCH (r:ReviewDecision {{estate_id:$estate_id,review_decision_id:$rid}}), "
                f"(n:{label} {{estate_id:$estate_id,{key}:$id}}) "
                "MERGE (r)-[:REVIEWS {estate_id:$estate_id}]->(n)",
                estate_id=scope,
                rid=decision["review_decision_id"],
                id=identifier,
            ).consume()
            if label == "Policy":
                tx.run(
                    "MATCH (p:Policy {estate_id:$estate_id,policy_id:$id}) "
                    "SET p.status='APPROVED',p.review_status='APPROVED',p.reviewed_hash=$hash,"
                    "p.review_purpose='POC_DEMO',p.demo_reviewed_hash=$hash,p.demo_valid_from=$start,"
                    "p.demo_valid_to=$end,p.demo_review_batch_id=$batch,p.demo_review_decision_id=$rid",
                    estate_id=scope,
                    id=identifier,
                    hash=original["version_hash"],
                    start=receipt["demo_valid_from"],
                    end=receipt["demo_valid_to"],
                    batch=receipt["batch_id"],
                    rid=decision["review_decision_id"],
                ).consume()
            elif label == "FAQ":
                tx.run(
                    "MATCH (f:FAQ {estate_id:$estate_id,faq_id:$id}) "
                    "SET f.demo_review_status='APPROVED',f.demo_reviewed_hash=$hash,"
                    "f.demo_valid_from=$start,f.demo_valid_to=$end,f.demo_review_batch_id=$batch,"
                    "f.demo_review_decision_id=$rid",
                    estate_id=scope, id=identifier, hash=original["version_hash"],
                    start=receipt["demo_valid_from"], end=receipt["demo_valid_to"],
                    batch=receipt["batch_id"], rid=decision["review_decision_id"],
                ).consume()
            else:
                tx.run(
                    f"MATCH (n:{label} {{estate_id:$estate_id,{key}:$id}}) "
                    "SET n.demo_verification_status='VERIFIED',n.demo_reviewed_hash=$hash,"
                    "n.demo_review_batch_id=$batch,n.demo_review_decision_id=$rid",
                    estate_id=scope,
                    id=identifier,
                    hash=original["version_hash"],
                    batch=receipt["batch_id"],
                    rid=decision["review_decision_id"],
                ).consume()
            existing_decisions += 1
        return existing_decisions

    # A single graph transaction publishes both estates atomically; queries inside it are all scoped.
    with repo.driver.session(database=repo.settings.database) as session:
        decision_count = session.execute_write(transaction)
    with auth.connect() as db:
        for scope in scopes:
            db.execute(
                "UPDATE human_review_receipts SET state='PUBLISHED' WHERE estate_id=? AND batch_id=?",
                (scope, receipt["batch_id"]),
            )
    return {
        "batch_id": receipt["batch_id"],
        "review_purpose": "POC_DEMO",
        "human_receipt_hash": receipt["receipt_hash"],
        "approved_policy_counts": {s: sum(label == "Policy" and p["estate_id"] == s for label, p in records) for s in scopes},
        "approved_faq_counts": {s: sum(label == "FAQ" and p["estate_id"] == s for label, p in records) for s in scopes},
        "evidence_chunks_verified_for_demo": sum(label == "EvidenceChunk" for label, _ in records),
        "documents_verified_for_demo": sum(label == "Document" for label, _ in records),
        "review_decisions": decision_count,
        "demo_valid_from": receipt["demo_valid_from"],
        "real_policy_dates_changed": False,
        "original_review_queue_changed": False,
    }
