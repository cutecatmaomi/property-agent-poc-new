"""Read FAQ knowledge only with current, exact-version human approvals and replayable XLSX evidence."""

import json
from datetime import date
from pathlib import Path

from .domain import canonical, digest
from .faq_contract import FAQRecord, xlsx_chunk_identity
from .knowledge import _review_matches
from .qa_supplement import extract_xlsx

DEMO_BLOCKING_ISSUES = frozenset({
    "BASELINE_CONFLICT_REVIEW_REQUIRED", "SAME_QUESTION_DIFFERENT_ANSWERS",
    "TRUNCATED_OR_PLACEHOLDER_CONTENT", "MULTIPLE_TOPICS_IN_ONE_CELL", "QUESTION_CONTEXT_MISSING",
    "SAFETY_CLAIM_REQUIRES_QUALIFIED_REVIEW", "LOCATION_OR_PROJECT_CONTEXT_NEEDS_REVIEW",
})


def confirmed_faq(bundle, scope, day, root, reviewer_authorized, unit=None, building=None, renovation_activity=None):
    return _faq_eligible(bundle, scope, day, root, reviewer_authorized, unit, building, renovation_activity, "CURRENT_POLICY")


def demo_faq(bundle, scope, day, root, reviewer_authorized, unit=None, building=None, renovation_activity=None):
    return _faq_eligible(bundle, scope, day, root, reviewer_authorized, unit, building, renovation_activity, "POC_DEMO")


def verified_demo_faq_source(bundle, scope, day, root, reviewer_authorized, unit=None, building=None, renovation_activity=None):
    """Verify source/content approval only. A separate authorized answer selection is required to use it."""
    return _faq_eligible(bundle, scope, day, root, reviewer_authorized, unit, building, renovation_activity, "POC_DEMO", require_issue_clearance=False)


def _faq_eligible(bundle, scope, day, root, reviewer_authorized, unit, building, renovation_activity, purpose, *, require_issue_clearance=True):
    try:
        stored = bundle["faq"]
        record = json.loads(stored["_record_json"])
        for key, value in record.items():
            if key in {"review_status", "reviewed_hash", "review_decision_id", "runtime_imported"}:
                continue
            projected = stored.get(key + "_json") if isinstance(value, (list, dict)) else stored.get(key)
            if projected != (canonical(value) if isinstance(value, (list, dict)) else value):
                return False
        for key in ("review_status", "reviewed_hash", "review_decision_id", "runtime_imported"):
            record[key] = stored.get(key)
        faq = FAQRecord.model_validate(record).model_dump()
        if faq["estate_id"] != scope:
            return False
        batch = None
        if purpose == "CURRENT_POLICY":
            if faq["review_status"] != "APPROVED":
                return False
            if date.fromisoformat(faq["valid_from"]) > day or (faq["valid_to"] and date.fromisoformat(faq["valid_to"]) < day):
                return False
            decision_id = faq["review_decision_id"]
        else:
            if (faq["review_status"] != "PENDING_REVIEW" or stored.get("demo_review_status") != "APPROVED"
                or stored.get("demo_reviewed_hash") != faq["version_hash"]
                or not stored.get("demo_review_batch_id") or not stored.get("demo_valid_from")
                or (require_issue_clearance and DEMO_BLOCKING_ISSUES.intersection(faq["blocking_issues"]))):
                return False
            if date.fromisoformat(stored["demo_valid_from"]) > day or (stored.get("demo_valid_to") and date.fromisoformat(stored["demo_valid_to"]) < day):
                return False
            batch, decision_id = stored["demo_review_batch_id"], stored.get("demo_review_decision_id")
        conditions = faq["applicable_to"]
        for key in ("residence_category", "has_elevator"):
            if conditions[key] is not None and (not unit or unit.get(key) != conditions[key]):
                return False
        if conditions["building_form"] is not None and (not building or building.get("building_form") != conditions["building_form"]):
            return False
        if conditions["renovation_activity"] not in (None, renovation_activity):
            return False
        if not any(r.get("review_decision_id") == decision_id and _review_matches(
            r, scope, "FAQ", faq["faq_id"], faq["version_hash"], purpose, day, reviewer_authorized, batch
        ) for r in bundle["reviews"]):
            return False
        chunk, document = bundle["chunk"], bundle["document"]
        for node, label, key, reviews in ((chunk, "EvidenceChunk", "chunk_id", bundle["chunk_reviews"]),
                                           (document, "Document", "document_id", bundle["document_reviews"])):
            verification_key = "verification_status" if purpose == "CURRENT_POLICY" else "demo_verification_status"
            if node.get("estate_id") != scope or node.get(verification_key) != "VERIFIED":
                return False
            if purpose == "POC_DEMO" and node.get("demo_reviewed_hash") != node.get("version_hash"):
                return False
            if not any((purpose != "POC_DEMO" or r.get("review_decision_id") == node.get("demo_review_decision_id"))
                       and _review_matches(r, scope, label, node[key], node["version_hash"], purpose, day, reviewer_authorized, batch)
                       for r in reviews):
                return False
        if document.get("document_id") != faq["document_id"] or chunk.get("document_id") != faq["document_id"] or chunk.get("chunk_id") != faq["chunk_id"]:
            return False
        root = Path(root).resolve()
        path = (root / faq["source_file"]).resolve()
        if not path.is_relative_to(root / "data/raw/shared_qa") or document.get("source_path") != faq["source_file"]:
            return False
        if not (digest(path.read_bytes()) == faq["source_sha256"] == document.get("sha256") == document.get("version_hash")):
            return False
        row = next(r for r in extract_xlsx(path) if (r["sheet"], r["row"]) == (faq["source_locator"]["sheet"], faq["source_locator"]["row"]))
        if (row["question"], row["answer"], row["work_order_flag"], row["work_order_type"]) != (
            faq["source_question"], faq["source_answer"], faq["source_work_order"]["flag"], faq["source_work_order"]["type"]):
            return False
        _, expected_id, expected_hash, text = xlsx_chunk_identity(scope, faq["source_sha256"], faq["source_locator"], row["question"], row["answer"])
        return (chunk["chunk_id"] == expected_id and chunk["version_hash"] == expected_hash
                and chunk.get("text") == text and chunk.get("content_hash") == digest(text.encode())
                and chunk.get("source_sha256") == faq["source_sha256"]
                and chunk.get("source_locator_json") == canonical(faq["source_locator"])
                and chunk.get("extraction_quality") == "NATIVE_TEXT_EXTRACTED")
    except (ValueError, TypeError, KeyError, OSError, StopIteration):
        return False
