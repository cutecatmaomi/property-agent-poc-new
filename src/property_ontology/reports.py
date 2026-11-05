"""Offline Day 1 deliverables. Human notes never promote candidate status."""

import json

from .models import OntologySnapshot

TOPICS = {"service": "物业服务", "fee": "收费", "decoration": "装修", "repair": "维修与保修"}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def source_location(chunk):
    loc = json.loads(chunk["locator_json"])
    if "page" in loc:
        label = f"{'渲染页' if loc['kind'] == 'docx_rendered_page' else 'PDF物理页'} {loc['page']}"
        if "table" in loc:
            label += f"，表 {loc['table']}，行 {loc['row']}"
        return label
    return "；".join(
        dict.fromkeys(
            f"正文元素 {s['body_child']}"
            + (f"，行{s['row']}列{s['column']}段{s['paragraph']}" if "row" in s else "")
            for s in loc.get("segments", [])
        )
    )


def build_day1_reports(settings):
    docs = settings.root / "docs"
    docs.mkdir(exist_ok=True)
    review_dir = settings.root / "data/day1"
    review_dir.mkdir(exist_ok=True)
    previous_path = review_dir / "review_queue.json"
    previous = (
        {r["policy"]["id"]: r.get("human_review", {}) for r in json.loads(previous_path.read_text())}
        if previous_path.exists()
        else {}
    )
    inventory, queue = [], []
    for estate in ("E001", "E002"):
        base = settings.root / "data/derived" / estate
        if not (base / "documents.jsonl").exists():
            continue
        documents = {d["id"]: d for d in read_jsonl(base / "documents.jsonl")}
        chunks = {c["id"]: c for c in read_jsonl(base / "evidence.jsonl")}
        candidates = read_jsonl(base / "candidates.jsonl")
        inventory.extend(documents.values())
        for candidate in candidates:
            chunk = chunks[candidate["chunk_id"]]
            document = documents[candidate["document_id"]]
            assert candidate["estate_id"] == chunk["estate_id"] == document["estate_id"] == estate
            assert candidate["status"] == "PENDING_REVIEW"
            assert candidate["statement"] == chunk["text"]
            queue.append(
                {
                    "review_ref": f"{estate}-K{len([r for r in queue if r['policy']['estate_id'] == estate]) + 1:02}",
                    "policy": candidate,
                    "document": document,
                    "evidence": chunk,
                    "source_location": source_location(chunk),
                    "exact_evidence_match": True,
                    "human_review": previous.get(
                        candidate["id"],
                        {
                            "reviewer": "",
                            "decision": "",
                            "applicability_notes": "",
                            "valid_from": "",
                            "valid_to": "",
                            "reviewed_at": "",
                        },
                    ),
                }
            )
    (docs / "file_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n")
    previous_path.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n")
    schema = settings.root / "ontology"
    schema.mkdir(exist_ok=True)
    # Legacy extraction schema must not replace the Day 2 ten-entity runtime contract.
    (schema / "day1.schema.json").write_text(
        json.dumps(OntologySnapshot.model_json_schema(), ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "# Day 1 待审核知识表",
        "",
        "本表为自动提取的候选，不是已确认政策。候选全部为 PENDING_REVIEW。",
        "人工记录可填写 data/day1/review_queue.json 的 human_review 字段；重建报告保留相同候选ID的记录，不改变候选审批状态。",
        "",
        "| 编号 | 楼盘 | 主题 | 原始文件 | 提取质量 | 状态 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in queue:
        p, d, c = row["policy"], row["document"], row["evidence"]
        lines.append(
            f"| {row['review_ref']} | {p['estate_id']} | {TOPICS[p['topic']]} | {d['filename']} | {c['quality']} | PENDING_REVIEW |"
        )
    for row in queue:
        p, d, c = row["policy"], row["document"], row["evidence"]
        lines.extend(
            [
                "",
                f"## {row['review_ref']} {TOPICS[p['topic']]}",
                "",
                f"来源：{d['source_path']}；{row['source_location']}",
                "",
                f"文档 SHA-256：`{d['sha256']}`",
                "",
                f"候选 ID：`{p['id']}`",
                "",
                "原文证据：",
                "",
                *["> " + line for line in p["statement"].splitlines()],
                "",
                f"适用条件：{p['conditions']}",
                "",
                "待核验事项：" + "、".join(json.loads(d.get("issues_json", "[]"))),
                "",
                "人工审核人：未确认；审核结论：未确认；政策状态：PENDING_REVIEW。",
            ]
        )
    (docs / "knowledge_review.md").write_text("\n".join(lines) + "\n")
    return {"documents": len(inventory), "candidates": len(queue), "approved": 0}
