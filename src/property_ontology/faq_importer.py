"""Scoped, source-replayed pending FAQ import. Does not approve knowledge or grant reviewer rights."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from .domain import digest
from .faq_contract import xlsx_chunk_identity
from .importer import properties
from .qa_supplement import extract_xlsx, normalized, validate_prepared_candidate

FAQ_PACKET = "data/day1/qa_supplement_20261029"


def load_faq_packet(root):
    root = Path(root).resolve()
    packet = root / FAQ_PACKET
    candidates = json.loads((packet / "candidates.json").read_text(encoding="utf-8"))
    manifest = json.loads((packet / "source_manifest.json").read_text(encoding="utf-8"))
    source = (root / manifest["path"]).resolve()
    if not source.is_relative_to(root / "data/raw/shared_qa") or digest(source.read_bytes()) != manifest["sha256"]:
        raise ValueError("FAQ原件路径或字节哈希不一致")
    rows = extract_xlsx(source)
    byrow = {(r["sheet"], r["row"]): r for r in rows}
    if not candidates or len({c["faq_id"] for c in candidates}) != len(candidates):
        raise ValueError("FAQ候选为空或标识重复")
    for c in candidates:
        validate_prepared_candidate(c)
        if c["source_file"] != manifest["path"] or c["source_sha256"] != manifest["sha256"]:
            raise ValueError("FAQ候选原件不属于本批")
        actual = byrow[c["source_locator"]["sheet"], c["source_locator"]["row"]]
        if (c["source_question"], c["source_answer"], c["source_work_order"]) != (
            actual["question"], actual["answer"], {"flag": actual["work_order_flag"], "type": actual["work_order_type"]}
        ):
            raise ValueError("FAQ问题、答案或原工单提示无法回放")
        for n in c["source_rows"]:
            alias = byrow[c["source_locator"]["sheet"], n]
            if any(normalized(alias[k]) != normalized(actual[k]) for k in
                   ("question", "answer", "work_order_flag", "work_order_type")):
                raise ValueError("FAQ合并来源行内容不一致")
    return candidates, rows


def faq_nodes_and_edges(candidates, rows):
    """Materialize every referenced node; the shared source still gets scoped document/chunk IDs."""
    nodes = defaultdict(dict)
    edges = []
    for c in candidates:
        scope = c["estate_id"]
        doc_id, chunk_id, chunk_version, text = xlsx_chunk_identity(
            scope, c["source_sha256"], c["source_locator"], c["source_question"], c["source_answer"]
        )
        document = {
            "id": doc_id, "document_id": doc_id, "estate_id": scope, "source_path": c["source_file"],
            "filename": Path(c["source_file"]).name, "source_format": "XLSX",
            "sha256": c["source_sha256"], "version_hash": c["source_sha256"],
            "last_parsed_at": "2026-10-29", "parser_version": c["contract_version"],
            "parse_status": "NATIVE_TEXT_EXTRACTED", "verification_status": "PENDING_VERIFICATION",
            "text_chars": sum(len(r["question"] + r["answer"]) for r in rows),
            "issues_json": '["HUMAN_REVIEW_REQUIRED"]', "is_mock": False, "data_kind": "REAL_DOCUMENT",
        }
        chunk = {
            "chunk_id": chunk_id, "estate_id": scope, "document_id": doc_id,
            "source_path": c["source_file"], "source_sha256": c["source_sha256"], "text": text,
            "content_hash": digest(text.encode()), "version_hash": chunk_version,
            "source_locator": c["source_locator"], "extraction_method": "native_xlsx",
            "extraction_version": c["contract_version"], "extraction_quality": "NATIVE_TEXT_EXTRACTED",
            "verification_status": "PENDING_VERIFICATION", "original_candidate_ids": [],
            "original_chunk_ids": [], "is_mock": False, "data_kind": "REAL_DOCUMENT_EVIDENCE",
        }
        service = {
            "estate_id": scope, "service_id": c["service_id"], "service_code": c["service_code"],
            "name": c["category"], "enabled": True, "capability_status": "KNOWLEDGE_ONLY",
            "is_mock": False, "data_kind": "SERVICE_TAXONOMY",
        }
        for label, key, row in [("Document", "document_id", document), ("EvidenceChunk", "chunk_id", chunk),
                                ("Service", "service_id", service), ("FAQ", "faq_id", dict(c, runtime_imported=True))]:
            nodes[label][row[key]] = row
        for relation, source_type, source_id, target_type, target_id in [
            ("HAS_DOCUMENT", "Estate", scope, "Document", doc_id),
            ("HAS_CHUNK", "Document", doc_id, "EvidenceChunk", chunk_id),
            ("OFFERS", "Estate", scope, "Service", c["service_id"]),
            ("APPLIES_TO", "FAQ", c["faq_id"], "Estate", scope),
            ("GOVERNS", "FAQ", c["faq_id"], "Service", c["service_id"]),
            ("SUPPORTED_BY", "FAQ", c["faq_id"], "EvidenceChunk", chunk_id),
        ]:
            edges.append({"estate_id": scope, "relation": relation, "source_type": source_type,
                          "source_id": source_id, "target_type": target_type, "target_id": target_id})
    unique_edges = {tuple(e[k] for k in ("estate_id", "relation", "source_id", "target_id")): e for e in edges}
    return {label: list(items.values()) for label, items in nodes.items()}, list(unique_edges.values())


def import_faq_candidates(repo, root):
    candidates, rows = load_faq_packet(root)  # Validate every estate before the first write.
    nodes, edges = faq_nodes_and_edges(candidates, rows)
    for scope in ("E001", "E002"):
        repo.seed(scope, {label: [properties(r, label != "Service") for r in records if r["estate_id"] == scope]
                          for label, records in nodes.items()}, [r for r in edges if r["estate_id"] == scope])
    return {"imported_or_existing_pending_faqs": dict(Counter(c["estate_id"] for c in candidates)),
            "auto_approved": 0, "source_documents": len(nodes["Document"]), "source_chunks": len(nodes["EvidenceChunk"])}
