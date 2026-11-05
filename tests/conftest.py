"""Synthetic approved proof used ONLY in temporary unit fixtures / isolated test Neo4j."""

import json
from zipfile import ZipFile

import pytest

from property_ontology.config import ROOT
from property_ontology.domain import VERSION_FIELDS, digest
from property_ontology.importer import properties


@pytest.fixture
def approved_bundle(tmp_path):
    root = tmp_path / "synthetic-root"
    source = root / "data/raw/estate_a/test-only.docx"
    source.parent.mkdir(parents=True)
    text = "合成测试物业服务规则，严禁用于真实楼盘答复。"
    with ZipFile(source, "w") as z:
        z.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p><w:r><w:t>" + text + "</w:t></w:r></w:p></w:body></w:document>",
        )
    d = {
        "estate_id": "E001",
        "document_id": "E001:DOC:TEST-APPROVED",
        "source_path": "data/raw/estate_a/test-only.docx",
        "sha256": digest(source.read_bytes()),
        "verification_status": "VERIFIED",
    }
    d["version_hash"] = d["sha256"]
    locator = {
        "kind": "DOCX_XML",
        "parts": [
            {
                "paragraph_index": 1,
                "xml_path": "/w:document/w:body/w:p",
                "start_char": 0,
                "end_char": len(text),
            }
        ],
    }
    c = {
        "estate_id": "E001",
        "document_id": d["document_id"],
        "source_locator": locator,
        "text": text,
        "content_hash": digest(text.encode()),
        "extraction_method": "native_docx",
        "extraction_version": "test-only",
        "verification_status": "VERIFIED",
        "extraction_quality": "NATIVE_TEXT_EXTRACTED",
    }
    identity = [c["document_id"], locator, c["content_hash"], c["extraction_method"], c["extraction_version"]]
    c["version_hash"] = digest(identity)
    c["chunk_id"] = "E001:CHUNK:" + digest(identity)
    p = json.loads((ROOT / "outputs/schema_alignment_20261027/data/policies.json").read_text())[0]
    p.update(
        policy_family_id="E001:PF:" + digest(["E001", "test.synthetic"]),
        service_id="E001:SVC:property_service",
        service_code="property_service",
        topic="rule",
        title="合成测试物业规则",
        question="合成物业测试规则？",
        statement=text,
        scope_note="仅供自动化测试",
        source_file=d["source_path"],
        source_locator=locator,
        source_excerpt=text,
        document_id=d["document_id"],
        chunk_ids=[c["chunk_id"]],
        status="APPROVED",
        review_status="APPROVED",
        valid_from="2026-01-01",
        valid_to="2027-12-31",
        validity_status="CONFIRMED",
        conflict_status="CLEARED",
        blocking_issues=[],
    )
    p["conditions"] = {
        **p["conditions"],
        "description": "仅供自动化测试",
        "validity": "CONFIRMED",
        "current_applicability": "CONFIRMED",
        "qualifiers": [],
    }
    p["fact"] = {
        "fact_key": "test.synthetic",
        "subject": "test_rule",
        "predicate": "rule",
        "value": text,
        "unit": None,
        "source_anchors": [text],
        "value_role": "STANDALONE",
        "component_of_fact_key": None,
    }
    p["provenance"] = [
        {
            "chunk_id": c["chunk_id"],
            "document_id": d["document_id"],
            "document_sha256": d["sha256"],
            "excerpt_sha256": c["content_hash"],
        }
    ]
    p["version_hash"] = digest({k: p[k] for k in VERSION_FIELDS})
    p["reviewed_hash"] = p["version_hash"]
    p["policy_id"] = "E001:POL:" + digest([p["policy_family_id"], p["version_hash"]])
    r = {
        "estate_id": "E001",
        "review_decision_id": "E001:REVIEW:TEST-ONLY",
        "actor_ref": "E001:REVIEWER:TEST-ONLY",
        "actor_role": "HUMAN_REVIEWER",
        "authorization_ref": "E001:GRANT:TEST-ONLY",
        "target_type": "Policy",
        "target_id": p["policy_id"],
        "target_version_hash": p["version_hash"],
        "decision": "APPROVED",
        "note": "独立临时测试库中的合成审核事件，不是任何真实知识的批准",
        "decided_at": "2026-01-01T00:00:00+00:00",
        "is_mock": False,
    }
    r["version_hash"] = digest({k: v for k, v in r.items() if k != "review_decision_id"})
    bundle = {
        "policy": properties(p, True),
        "evidence": [{"chunk": properties(c, True), "document": properties(d, True)}],
        "reviews": [r],
    }
    # Ensure no accidental mutable alias to the original alignment packet.
    assert json.loads(bundle["policy"]["_record_json"])["statement"] == text
    return root, bundle
