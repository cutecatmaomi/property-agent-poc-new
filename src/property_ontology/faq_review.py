"""Read human worksheet proposals and generate pending revisions; never grants approval."""

import json
from datetime import date, timedelta
from pathlib import PurePosixPath
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from .faq_contract import CATEGORY_LABELS, prepare_review_revision
from .faq_importer import load_faq_packet

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def review_cells(path):
    with ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        props = workbook.find("s:workbookPr", NS)
        if props is not None and props.get("date1904") in {"1", "true"}:
            raise ValueError("本审核模板只接受1900日期系统")
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(t.text or "" for t in x.findall(".//s:t", NS))
                       for x in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        links = {r.get("Id"): r.get("Target") for r in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
        sheet = next(s for s in workbook.findall("s:sheets/s:sheet", NS) if s.get("name") == "补充审核")
        target = links[sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")]
        target = target.lstrip("/") if target.startswith("/") else str(PurePosixPath("xl") / target)
        cells = {}
        for c in ET.fromstring(archive.read(target)).findall(".//s:sheetData/s:row/s:c", NS):
            value = c.find("s:v", NS)
            value = value.text or "" if value is not None else ""
            if c.get("t") == "s" and value:
                value = strings[int(value)]
            elif c.get("t") == "inlineStr":
                value = "".join(t.text or "" for t in c.findall("s:is//s:t", NS))
            cells[c.get("r")] = (value, c.get("t"), c.find("s:f", NS) is not None)
        return cells


def _date_cell(cell):
    value, kind, formula = cell
    if formula:
        raise ValueError("有效期必须由人工填写日期，不能使用公式")
    if not value:
        return None
    if kind in {None, "n"}:
        serial = float(value)
        if not serial.is_integer() or serial < 61:
            raise ValueError("有效期日期数值无效")
        return (date(1899, 12, 30) + timedelta(days=int(serial))).isoformat()
    return date.fromisoformat(value[:10]).isoformat()


def prepare_workbook_revisions(root, path):
    candidates, _ = load_faq_packet(root)
    originals = {c["faq_id"]: c for c in candidates}
    cells = review_cells(path)

    def value(col, row):
        cell = cells.get(f"{col}{row}", ("", None, False))
        if cell[2]:
            raise ValueError("人工意见字段不接受公式")
        return cell[0]

    if value("U", 6) != "FAQ唯一ID" or value("Y", 6) != "来源核验意见":
        raise ValueError("请使用Day2对齐后的审核模板")
    revisions, seen = [], set()
    for row in sorted(int(key[1:]) for key in cells if key.startswith("U") and key[1:].isdigit() and int(key[1:]) >= 7):
        key = value("U", row)
        if key not in originals or key in seen:
            raise ValueError("审核表引用未知、过时或重复FAQ版本")
        seen.add(key)
        c = originals[key]
        for col, expected in {"B": c["estate_id"], "C": CATEGORY_LABELS[c["category"]], "D": c["question"], "E": c["answer_candidate"],
                              "V": c["version_hash"], "M": c["review_status"], "N": c["scope_status"],
                              "X": c["source_sha256"], "W": c["chunk_id"], "AB": c["service_code"],
                              "AC": c["service_id"], "AD": c["verification_status"], "AE": c["document_id"],
                              "AF": c["source_file"], "O": f'{c["source_locator"]["sheet"]}!A{c["source_locator"]["row"]}:D{c["source_locator"]["row"]}',
                              "P": f'{c["source_work_order"]["flag"]} / {c["source_work_order"]["type"] or "未填写"}'}.items():
            if value(col, row) != expected:
                raise ValueError("原始版本/状态被编辑；只填写黄色人工意见列")
        if json.loads(value("AG", row)) != c["applicable_to"]:
            raise ValueError("结构化适用条件是只读映射；条件变更需生成新候选版本")
        proposal = {
            "suggestion": value("G", row), "reviewer": value("H", row), "note": value("K", row),
            "confirmed_conditions": value("I", row), "proposed_answer": value("J", row),
            "proposed_valid_from": _date_cell(cells.get(f"S{row}", ("", None, False))),
            "proposed_valid_to": _date_cell(cells.get(f"T{row}", ("", None, False))),
            "source_check": value("Y", row), "scope_check": value("Z", row), "conflict_check": value("AA", row),
        }
        if proposal == c["human_review"]:
            continue
        revised = prepare_review_revision(c, proposal)
        revisions.append({"original_faq_id": key, "original_version_hash": c["version_hash"], "candidate": revised})
    if seen != set(originals):
        raise ValueError("审核模板记录缺失，不能作为完整批次处理")
    return {"format": "faq-review-proposals-v1", "reviewed_rows": len(seen), "revisions": revisions,
            "auto_approved": 0, "database_writes": 0}
