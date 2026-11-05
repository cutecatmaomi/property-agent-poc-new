"""Presentation metadata shared by Day1 exports and the Day2 source-selection contract."""

import json

from .answer_selection import PACKET, load_selections
from .faq_demo_review import validate_faq_demo_receipt


def augment_day1_result(result, root):
    result["answer_selections"] = {}
    if (root / PACKET).exists():
        result["answer_selections"] = {r["faq_id"]: r for r in load_selections(root)}
    for pointer_name, result_key in (("source_priority_current.json", "source_priority_review"),
                                     ("faq_demo_current.json", "faq_demo_review")):
        path = root / "data/reviews" / pointer_name
        if not path.exists():
            continue
        pointer = json.loads(path.read_text())
        receipt = json.loads((root / pointer["receipt"]).read_text())
        if result_key == "faq_demo_review":
            validate_faq_demo_receipt(root, receipt, pointer["receipt_hash"])
        else:
            from .answer_selection import validate_selection_receipt
            validate_selection_receipt(root, receipt, pointer["receipt_hash"])
        result[result_key] = {"batch_id": receipt["batch_id"], "receipt_hash": receipt["receipt_hash"],
                              "valid_from": receipt["demo_valid_from"], "targets": receipt["targets"],
                              "published": (root / "data/reviews" / (receipt["batch_id"] + ".published.json")).exists()}
    return result
