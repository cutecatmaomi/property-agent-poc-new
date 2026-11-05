"""Source/approval gate for future consultation. No pending statement is returned to residents."""

import json
from datetime import date, datetime
from pathlib import Path

from .domain import VERSION_FIELDS, PolicyRecord, ReviewDecisionRecord, digest, scoped


def confirmed_policy(
    bundle, scope, day, root, reviewer_authorized, unit=None, building=None, renovation_activity=None
):
    return _policy_eligible(
        bundle,
        scope,
        day,
        root,
        reviewer_authorized,
        unit,
        building,
        purpose="CURRENT_POLICY",
        renovation_activity=renovation_activity,
    )


def demo_policy(
    bundle, scope, day, root, reviewer_authorized, unit=None, building=None, renovation_activity=None
):
    return _policy_eligible(
        bundle,
        scope,
        day,
        root,
        reviewer_authorized,
        unit,
        building,
        purpose="POC_DEMO",
        renovation_activity=renovation_activity,
    )


def _review_matches(
    row, scope, target_type, target_id, target_hash, purpose, day, reviewer_authorized, batch=None, component="CONTENT"
):
    try:
        review = ReviewDecisionRecord.model_validate(row)
        return (
            review.estate_id == scope
            and review.target_type == target_type
            and review.target_component == component
            and review.target_id == target_id
            and review.target_version_hash == target_hash
            and review.review_purpose == purpose
            and review.decision == ("APPROVED" if target_type in {"Policy", "FAQ"} else "VERIFIED")
            and not review.is_mock
            and datetime.fromisoformat(review.decided_at).date() <= day
            and (not batch or review.review_batch_id == batch)
            and reviewer_authorized(scope, review.actor_ref, review.authorization_ref)
        )
    except (ValueError, TypeError):
        return False


def _policy_eligible(
    bundle, scope, day, root, reviewer_authorized, unit, building, *, purpose, renovation_activity
):
    """Fail closed on missing evidence, dates, authorization, hashes or scope information."""
    try:
        stored = bundle["policy"]
        record = json.loads(stored["_record_json"])
        for field in ["estate_id", "policy_id", "service_id", "service_code", "version_hash"]:
            if stored.get(field) != record.get(field):
                return False
        # Approval metadata is a projection; immutable semantic payload stays bound to its version.
        for key in [
            "status",
            "review_status",
            "reviewed_hash",
            "valid_from",
            "valid_to",
            "validity_status",
            "conflict_status",
        ]:
            record[key] = stored.get(key)
        record["blocking_issues"] = json.loads(stored.get("blocking_issues_json", "null"))
        p = PolicyRecord.model_validate(record).model_dump()
        if p["estate_id"] != scope or p["status"] != "APPROVED" or p["review_status"] != "APPROVED":
            return False
        if p["reviewed_hash"] != p["version_hash"] or p["version_hash"] != digest(
            {k: p[k] for k in VERSION_FIELDS}
        ):
            return False
        if purpose == "CURRENT_POLICY":
            if (
                p["validity_status"] != "CONFIRMED"
                or p["conflict_status"] != "CLEARED"
                or p["blocking_issues"]
            ):
                return False
            if (
                p["conditions"]["validity"] != "CONFIRMED"
                or p["conditions"]["current_applicability"] != "CONFIRMED"
            ):
                return False
            if not p["valid_from"] or date.fromisoformat(p["valid_from"]) > day:
                return False
            if p["valid_to"] and date.fromisoformat(p["valid_to"]) < day:
                return False
        else:
            if (
                stored.get("review_purpose") != "POC_DEMO"
                or not stored.get("demo_review_batch_id")
                or stored.get("demo_reviewed_hash") != p["version_hash"]
            ):
                return False
            if not stored.get("demo_valid_from") or date.fromisoformat(stored["demo_valid_from"]) > day:
                return False
            if stored.get("demo_valid_to") and date.fromisoformat(stored["demo_valid_to"]) < day:
                return False
        targets = p["conditions"]["applicable_to"]
        if (
            p["service_code"] == "property_fee"
            and p["topic"] == "fee"
            and targets.get("residence_category") in (None, "UNSPECIFIED")
        ):
            return False
        for key in ["residence_category", "has_elevator"]:
            value = targets.get(key)
            if value not in (None, "UNSPECIFIED") and (not unit or unit.get(key) != value):
                return False
        if targets.get("building_form") not in (None, "UNSPECIFIED"):
            if not building or building.get("building_form") != targets["building_form"]:
                return False
        if targets.get("renovation_activity") not in (None, "UNSPECIFIED", renovation_activity):
            return False
        authorized = any(
            _review_matches(
                r,
                scope,
                "Policy",
                p["policy_id"],
                p["version_hash"],
                purpose,
                day,
                reviewer_authorized,
                stored.get("demo_review_batch_id") if purpose == "POC_DEMO" else None,
            )
            for r in bundle["reviews"]
            if r
        )
        if not authorized:
            return False
        chunks = {x["chunk"]["chunk_id"]: x for x in bundle["evidence"]}
        if set(chunks) != set(p["chunk_ids"]):
            return False
        root = Path(root).resolve()
        for proof in p["provenance"]:
            c, d = chunks[proof["chunk_id"]]["chunk"], chunks[proof["chunk_id"]]["document"]
            if any(x.get("estate_id") != scope for x in (c, d)):
                return False
            if purpose == "CURRENT_POLICY":
                if any(x.get("verification_status") != "VERIFIED" for x in (c, d)):
                    return False
            else:
                entry = chunks[proof["chunk_id"]]
                for node, kind, key, review_field in [
                    (c, "EvidenceChunk", "chunk_id", "chunk_review"),
                    (d, "Document", "document_id", "document_review"),
                ]:
                    if (
                        node.get("demo_verification_status") != "VERIFIED"
                        or node.get("demo_reviewed_hash") != node.get("version_hash")
                        or not _review_matches(
                            entry.get(review_field),
                            scope,
                            kind,
                            node[key],
                            node["version_hash"],
                            "POC_DEMO",
                            day,
                            reviewer_authorized,
                        )
                    ):
                        return False
            if c.get("extraction_quality") != "NATIVE_TEXT_EXTRACTED":
                # This Day 2 reader has no approved OCR/transcription import path.
                return False
            if d.get("document_id") != proof["document_id"] or c.get("document_id") != d.get("document_id"):
                return False
            if d.get("version_hash") != d.get("sha256") or d["sha256"] != proof["document_sha256"]:
                return False
            path = (root / d["source_path"]).resolve()
            if not path.is_relative_to(root / "data/raw") or digest(path.read_bytes()) != d["sha256"]:
                return False
            text_hash = digest(c["text"].encode())
            if c.get("content_hash") != text_hash or proof["excerpt_sha256"] != text_hash:
                return False
            locator = json.loads(c["source_locator_json"])
            identity = [c["document_id"], locator, text_hash, c["extraction_method"], c["extraction_version"]]
            if c.get("version_hash") != digest(identity) or not scoped(c["chunk_id"], scope):
                return False
            if c["chunk_id"] != f"{scope}:CHUNK:{digest(identity)}":
                return False
        return True
    except (KeyError, ValueError, TypeError, OSError):
        return False
