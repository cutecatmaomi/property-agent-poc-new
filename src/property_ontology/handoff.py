"""Restore version-bound Day 2 demo inputs into a recipient's own empty runtime."""

import hashlib
import json
from pathlib import Path

from .domain import CORE_IDS, RELATIONS, canonical, scoped

GROUPS = {"Building": "buildings", "Unit": "units", "Resident": "residents", "WorkOrder": "work_orders"}


def checked_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("交付清单路径越界")
    return path


def validate_manifest(root, manifest):
    root = Path(root).resolve()
    if manifest.get("format") != "day2-offline-handoff-v1":
        raise ValueError("未知交付清单格式")
    files = manifest["files"]
    for relative, expected in files.items():
        path = checked_path(root, relative)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("交付文件校验失败：" + relative)
    receipt_file = manifest["human_receipt"]["path"]
    if receipt_file not in files:
        raise ValueError("人工审核回执未被交付清单绑定")
    mocks = {}
    for scope in ("E001", "E002"):
        relative = manifest["mock_snapshots"][scope]
        if relative not in files:
            raise ValueError("Mock快照未被交付清单绑定")
        mock = json.loads(checked_path(root, relative).read_text(encoding="utf-8"))
        if mock["estate_id"] != scope or mock["data_kind"] != "LIVE_MOCK_BUSINESS_SNAPSHOT":
            raise ValueError("Mock快照楼盘或类型不正确")
        identifiers = set()
        for label, group in {"Estate": "estates", "Service": "services", **GROUPS}.items():
            for row in mock[group]:
                key = CORE_IDS[label]
                if row.get("estate_id") != scope or not scoped(row[key], scope):
                    raise ValueError("Mock快照包含跨楼盘节点")
                if label in GROUPS and row.get("is_mock") is not True:
                    raise ValueError("只允许恢复模拟业务对象")
                if row[key] in identifiers:
                    raise ValueError("Mock节点重复")
                identifiers.add(row[key])
        for edge in mock["relations"]:
            expected = RELATIONS.get(edge["relation"])
            if (
                not expected
                or edge["source_type"] != expected[0]
                or edge["target_type"] != expected[1]
                or edge["source_type"] not in {"Estate", "Service", *GROUPS}
                or edge["target_type"] not in {"Estate", "Service", *GROUPS}
                or edge["estate_id"] != scope
                or edge.get("properties", {}).get("estate_id", scope) != scope
                or edge["source_id"] not in identifiers
                or edge["target_id"] not in identifiers
            ):
                raise ValueError("Mock关系不属于当前楼盘受控业务对象")
        mocks[scope] = mock
    return mocks


def initialize_handoff(settings, manifest, *, human_ack_hash):
    # All hashes, IDs and permissions are checked before creating the first graph object/account.
    mocks = validate_manifest(settings.root, manifest)
    receipt = json.loads(
        checked_path(settings.root.resolve(), manifest["human_receipt"]["path"]).read_text(encoding="utf-8")
    )
    if human_ack_hash != manifest["human_receipt"]["ack_hash"]:
        raise ValueError("恢复已有人工审核需要匹配的完整确认哈希")
    from .actions import PropertyActions
    from .importer import import_knowledge, load_packet, properties, seed_mock
    from .review import publish_demo_review, validate_demo_receipt

    load_packet(settings.root)
    validate_demo_receipt(settings.root, receipt, human_ack_hash)
    app = PropertyActions(settings)
    try:
        app._repo.initialize_schema()
        imported = import_knowledge(app._repo, settings.root)
        seed_mock(app._repo, app.auth, settings.root)  # New locally generated passwords; no old sessions.
        for scope, mock in mocks.items():
            nodes = {label: [properties(row) for row in mock[group]] for label, group in GROUPS.items()}
            app._repo.seed(scope, nodes, mock["relations"])
            for row in app._repo._read(
                scope,
                lambda tx, estate_id=scope: tx.run(
                    "MATCH (w:WorkOrder {estate_id:$estate_id,is_mock:true}) RETURN properties(w) AS w",
                    estate_id=estate_id,
                ).data(),
            ):
                order = row["w"]
                app.auth.mirror_events(scope, order["work_order_id"], json.loads(order["state_history_json"]))
        review = publish_demo_review(
            app._repo, app.auth, settings.root, receipt, human_ack_hash=human_ack_hash
        )
        result = {
            "initialized": True,
            "import": imported,
            "human_review": review,
            "counts": {scope: app._repo.stats(scope) for scope in ("E001", "E002")},
            "credentials_file": str(app.auth.path.parent / "mock_credentials.json"),
            "old_sessions_restored": False,
        }
        path = app.auth.path.parent / "handoff_initialized.json"
        path.write_text(canonical(result) + "\n", encoding="utf-8")
        return result
    finally:
        app.close()
