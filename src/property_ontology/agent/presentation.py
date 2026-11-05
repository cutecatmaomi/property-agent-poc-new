"""Source-bound rendering. Knowledge answers are not generated or rewritten by a model."""

from ..day2_cli import knowledge_text

STATUS_NAMES = {"SUBMITTED": "已提交", "ASSIGNED": "已安排处理", "IN_PROGRESS": "处理中",
                "COMPLETED": "已完成", "CANCELLED": "已取消"}


def present(result):
    if "answer_text" in result:
        kind = {"ANSWERED": "KNOWLEDGE", "NEED_CLARIFICATION": "CLARIFY", "CONFLICT": "KNOWLEDGE_CONFLICT",
                "NOT_VERIFIED": "REVIEW_REQUIRED", "NOT_FOUND": "FIXED_REPLY"}[result["status"]]
        return {"kind": kind, "text": result["answer_text"], "data": result}
    if "response_type" in result:
        return {"kind": result["response_type"], "text": knowledge_text(result), "data": result}
    if "progress" in result:
        lines = ["工单编号：" + result["number"], "当前状态：" + STATUS_NAMES[result["status"]]]
        lines += [e["at"] + " · " + e["note"] for e in result["progress"]]
        return {"kind": "ORDER_STATUS", "text": "\n".join(lines), "data": result}
    if "number" in result:
        return {"kind": "ORDER_CREATED", "text": "报修已提交。工单编号：" + result["number"]
                + "\n当前状态：" + STATUS_NAMES[result["status"]], "data": result}
    return {"kind": result.get("kind", "CLARIFY"), "text": result["message"], "data": result}


def error_reply(exc):
    return {"kind": "ERROR", "text": exc.message, "data": {"error": exc.code, "status": exc.status}}
