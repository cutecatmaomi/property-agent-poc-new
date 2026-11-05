"""Validate the reviewed alignment packet, import only PENDING_REVIEW and seed mock-only business data."""

import json
import os
import re
import secrets
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import pdfplumber
from lxml import etree

from .domain import CORE_IDS, PolicyRecord, canonical, digest

PACKET = "outputs/schema_alignment_20261027"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def load_packet(root):
    root = Path(root)
    packet = root / PACKET
    data = {
        name: json.loads((packet / f"data/{name}.json").read_text())
        for name in [
            "policies",
            "evidence_chunks",
            "documents",
            "estates",
            "services",
            "relations",
            "lineage",
        ]
    }
    contract = json.loads((packet / "contract/data_contract.json").read_text())
    docs = {d["document_id"]: d for d in data["documents"]}
    chunks = {c["chunk_id"]: c for c in data["evidence_chunks"]}
    services = {s["service_id"]: s for s in data["services"]}
    if any(
        len({x[CORE_IDS[label]] for x in data[name]}) != len(data[name])
        for label, name in [
            ("Document", "documents"),
            ("EvidenceChunk", "evidence_chunks"),
            ("Policy", "policies"),
            ("Service", "services"),
        ]
    ):
        raise ValueError("知识包含重复主键")
    xml = {}
    pdf_tables = {}
    for d in docs.values():
        expected_dir = "estate_a" if d["estate_id"] == "E001" else "estate_b"
        path = (root / d["source_path"]).resolve()
        if not path.is_relative_to((root / f"data/raw/{expected_dir}").resolve()):
            raise ValueError("原始路径不属于目标楼盘")
        if digest(path.read_bytes()) != d["sha256"] or d["version_hash"] != d["sha256"]:
            raise ValueError("原始文件版本不一致")
        if d["verification_status"] != "PENDING_VERIFICATION":
            raise ValueError("导入器不接受自动核验后的文档")
    for c in chunks.values():
        d = docs[c["document_id"]]
        locator = c["source_locator"]
        if c["estate_id"] != d["estate_id"] or c["source_sha256"] != d["sha256"]:
            raise ValueError("证据归属/版本不一致")
        if c["verification_status"] != "PENDING_VERIFICATION":
            raise ValueError("导入器不批准证据")
        if locator["kind"] == "DOCX_XML":
            if d["document_id"] not in xml:
                with ZipFile(root / d["source_path"]) as z:
                    xml[d["document_id"]] = etree.fromstring(z.read("word/document.xml"))
            tree = xml[d["document_id"]]
            all_p = tree.findall(".//w:body//w:p", NS)
            parts = []
            for part in locator["parts"]:
                nodes = tree.xpath(part["xml_path"], namespaces=NS)
                if len(nodes) != 1 or all_p[part["paragraph_index"] - 1] is not nodes[0]:
                    raise ValueError("证据段落定位失效")
                text = "".join(nodes[0].xpath(".//w:t/text()", namespaces=NS))
                if not 0 <= part["start_char"] < part["end_char"] <= len(text):
                    raise ValueError("证据字符范围失效")
                parts.append(text[part["start_char"] : part["end_char"]])
            actual = "\n".join(parts)
        elif locator["kind"] == "PDF_TABLE":
            key = (d["document_id"], locator["page"], locator["table"])
            if key not in pdf_tables:
                with pdfplumber.open(root / d["source_path"]) as pdf:
                    table = pdf.pages[locator["page"] - 1].find_tables()[locator["table"] - 1]
                    pdf_tables[key] = (table.extract(), list(table.bbox))
            rows, bbox = pdf_tables[key]
            if len(bbox) != len(locator["bbox"]) or any(
                abs(a - b) > 0.001 for a, b in zip(bbox, locator["bbox"])
            ):
                raise ValueError("PDF表格位置失效")
            actual = "\n".join(" | ".join(str(cell or "") for cell in rows[n - 1]) for n in locator["rows"])
        else:
            raise ValueError("本批只接受可回放的原生证据；OCR继续待人工核验")
        identity = [
            c["document_id"],
            locator,
            digest(actual.encode()),
            c["extraction_method"],
            c["extraction_version"],
        ]
        if (
            actual != c["text"]
            or c["content_hash"] != digest(actual.encode())
            or c["version_hash"] != digest(identity)
        ):
            raise ValueError("证据原文或版本失效")
        if c["chunk_id"] != f"{c['estate_id']}:CHUNK:{digest(identity)}":
            raise ValueError("证据ID失效")
    old_ids = {r["original_candidate_id"]: r["estate_id"] for r in data["lineage"]}
    for p in data["policies"]:
        PolicyRecord.model_validate(p)
        if (
            p["status"] != "PENDING_REVIEW"
            or p["reviewed_hash"] is not None
            or p["valid_from"]
            or p["valid_to"]
        ):
            raise ValueError("知识导入不能批准政策、继承审核或推断有效期")
        if p["topic"] not in contract["service_codes"][p["service_code"]]["topics"]:
            raise ValueError("服务类别映射错误")
        if services[p["service_id"]]["estate_id"] != p["estate_id"]:
            raise ValueError("跨楼盘服务")
        expected = []
        texts = []
        for cid in p["chunk_ids"]:
            c = chunks[cid]
            if c["estate_id"] != p["estate_id"]:
                raise ValueError("跨楼盘证据")
            expected.append(
                {
                    "chunk_id": cid,
                    "document_id": c["document_id"],
                    "document_sha256": c["source_sha256"],
                    "excerpt_sha256": c["content_hash"],
                }
            )
            texts.append(c["text"])
        if p["provenance"] != expected or any(
            old_ids[o] != p["estate_id"] for o in p["original_candidate_ids"]
        ):
            raise ValueError("知识血缘错误")

        def normalize(s):
            return re.sub(r"\s+", "", s)

        if any(normalize(a) not in normalize("\n".join(texts)) for a in p["fact"]["source_anchors"]):
            raise ValueError("原文不能回放事实锚点")
    return data


def properties(row, immutable=False):
    props = {k: v for k, v in row.items() if v is None or isinstance(v, (str, bool, int, float))}
    for k, v in row.items():
        if isinstance(v, (dict, list)):
            props[k + "_json"] = canonical(v)
    if immutable:
        props["_record_json"] = canonical(row)
    return props


def import_knowledge(repo, root):
    data = load_packet(root)  # Validate all estates before the first transaction.
    mapping = {
        "Estate": "estates",
        "Service": "services",
        "Document": "documents",
        "EvidenceChunk": "evidence_chunks",
        "Policy": "policies",
    }
    result = {}
    for scope in ["E001", "E002"]:
        nodes = {
            label: [
                properties(r, label in {"Document", "EvidenceChunk", "Policy"})
                for r in data[name]
                if r["estate_id"] == scope
            ]
            for label, name in mapping.items()
        }
        repo.seed(scope, nodes, [r for r in data["relations"] if r["estate_id"] == scope])
        result[scope] = dict(Counter(p["estate_id"] for p in data["policies"]))[scope]
    return {"imported_or_existing_pending": result, "auto_approved": 0}


def seed_mock(repo, auth, root):
    root = Path(root)
    mock = json.loads((root / PACKET / "design/mock_instances.json").read_text())
    result = {}
    for scope in ["E001", "E002"]:
        nodes = defaultdict(list)
        for label, group in [
            ("Building", "buildings"),
            ("Unit", "units"),
            ("Resident", "residents"),
            ("WorkOrder", "work_orders"),
        ]:
            for original in mock[group]:
                if original["estate_id"] != scope:
                    continue
                if original.get("is_mock") is not True:
                    raise ValueError("业务实例必须是Mock")
                row = dict(original)
                if label == "Unit":
                    row["status"] = "ACTIVE"
                if label == "WorkOrder":
                    events = [
                        dict(e, public_note="模拟" + e["new_status"])
                        for e in mock["state_events"]
                        if e["work_order_id"] == row["work_order_id"]
                    ]
                    row["state_history_json"] = canonical(events)
                nodes[label].append(properties(row))
        repo.seed(scope, nodes, [r for r in mock["relations"] if r["estate_id"] == scope])
        for order in nodes["WorkOrder"]:
            # Mirror only after graph commit; authoritative history is committed atomically in Neo4j.
            auth.mirror_events(scope, order["work_order_id"], json.loads(order["state_history_json"]))
        result[scope] = {label: len(rows) for label, rows in nodes.items()}
    credential_path = auth.path.parent / "mock_credentials.json"
    credentials = json.loads(credential_path.read_text()) if credential_path.exists() else []
    if not credential_path.exists():
        with auth.connect() as db:
            for scope in ["E001", "E002"]:
                if db.execute("SELECT count(*) FROM accounts WHERE estate_id=?", (scope,)).fetchone()[0]:
                    raise ValueError("已有模拟账号但凭据文件缺失；保留数据库并恢复原mock_credentials.json")
    known = {(r["estate_id"], r["username"]) for r in credentials}
    accounts = []
    for scope in ["E001", "E002"]:
        for username, actor, role in [
            ("owner1", f"{scope}:RES:MOCK-01", "RESIDENT"),
            ("owner2", f"{scope}:RES:MOCK-02", "RESIDENT"),
            ("operator", f"{scope}:OP:MOCK-01", "BACKOFFICE_OPERATOR"),
        ]:
            if (scope, username) not in known:
                credentials.append(
                    {"estate_id": scope, "username": username, "password": secrets.token_urlsafe(20)}
                )
            record = next(x for x in credentials if x["estate_id"] == scope and x["username"] == username)
            accounts.append((scope, username, actor, role, record["password"]))
    descriptor = os.open(credential_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    if hasattr(os, "fchmod"):
        os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(json.dumps(credentials, ensure_ascii=False, indent=2))
    for args in accounts:
        auth.seed_account(*args)
    return {"mock_objects": result, "credentials_file": str(credential_path), "is_mock": True}
