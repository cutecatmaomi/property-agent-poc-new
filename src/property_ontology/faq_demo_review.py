"""Record the user's explicit Q&A confirmation for POC reference only; never publish current policies."""

from datetime import date, datetime
from pathlib import Path

from .domain import digest
from .faq_importer import faq_nodes_and_edges, load_faq_packet
from .review import publish_demo_records

RECEIPT_VERSION = "explicit-human-faq-demo-v1"


def build_faq_demo_receipt(root, *, statement, usage_confirmation, recorded_at, demo_from):
    candidates, _ = load_faq_packet(root)
    packet = Path(root) / "data/day1/qa_supplement_20261029/candidates.json"
    receipt = {
        "receipt_version": RECEIPT_VERSION, "review_purpose": "POC_DEMO",
        "human_origin": "USER_EXPLICIT_CONFIRMATION", "human_statement": statement,
        "human_usage_confirmation": usage_confirmation,
        "reviewer_label": "本项目对话用户（明确人工确认）", "recorded_at": recorded_at,
        "demo_valid_from": demo_from, "demo_valid_to": None, "real_policy_effective_dates_confirmed": False,
        "packet_file": str(packet.relative_to(root)), "packet_sha256": digest(packet.read_bytes()),
        "targets": [{"estate_id": c["estate_id"], "faq_id": c["faq_id"], "version_hash": c["version_hash"]} for c in candidates],
        "limitations": ["原件限定及来源保留", "真实政策日期、范围及冲突不因演示批准而消失",
                        "咨询工单和回电尚未实现", "同题异答、缺失截断、工程安全及原资料冲突仍阻断确定答复"],
    }
    receipt["batch_id"] = "faq-demo-" + digest(receipt)[:24]
    receipt["receipt_hash"] = digest(receipt)
    return receipt


def validate_faq_demo_receipt(root, receipt, human_ack_hash):
    if not human_ack_hash or human_ack_hash != receipt.get("receipt_hash"):
        raise ValueError("必须绑定已经明确给出的人工确认回执")
    if digest({k: v for k, v in receipt.items() if k != "receipt_hash"}) != human_ack_hash:
        raise ValueError("FAQ人工回执被修改")
    if (receipt.get("receipt_version") != RECEIPT_VERSION or receipt.get("review_purpose") != "POC_DEMO"
        or receipt.get("human_origin") != "USER_EXPLICIT_CONFIRMATION"
        or receipt.get("real_policy_effective_dates_confirmed") is not False
        or not receipt.get("human_statement") or not receipt.get("human_usage_confirmation")):
        raise ValueError("仅处理用户明确确认的FAQ演示审核")
    if datetime.fromisoformat(receipt["recorded_at"]).tzinfo is None:
        raise ValueError("审核时间必须有时区")
    start = date.fromisoformat(receipt["demo_valid_from"])
    if receipt["demo_valid_to"] and date.fromisoformat(receipt["demo_valid_to"]) < start:
        raise ValueError("演示有效期无效")
    root = Path(root).resolve()
    packet = root / "data/day1/qa_supplement_20261029/candidates.json"
    if receipt["packet_file"] != str(packet.relative_to(root)) or digest(packet.read_bytes()) != receipt["packet_sha256"]:
        raise ValueError("人工确认的FAQ批次版本已变化")
    candidates, rows = load_faq_packet(root)
    known = {c["faq_id"]: c for c in candidates}
    targets = receipt["targets"]
    if not targets or len({t["faq_id"] for t in targets}) != len(targets):
        raise ValueError("审核目标为空或重复")
    selected = []
    for t in targets:
        c = known.get(t["faq_id"])
        if not c or c["estate_id"] != t["estate_id"] or c["version_hash"] != t["version_hash"]:
            raise ValueError("FAQ审核目标与楼盘、内容版本不符")
        selected.append(c)
    nodes, _ = faq_nodes_and_edges(selected, rows)
    return [(label, c) for label in ("Document", "EvidenceChunk", "FAQ") for c in nodes[label]]


def publish_faq_demo_review(repo, auth, root, receipt, *, human_ack_hash):
    records = validate_faq_demo_receipt(root, receipt, human_ack_hash)
    return publish_demo_records(repo, auth, receipt, records)
