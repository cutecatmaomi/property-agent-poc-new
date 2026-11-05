"""Read-only operation receipts for dialogue; a fault report is not an action receipt."""

import copy


def operation_snapshot(session):
    pending = session._turns.get(session._pending, {}) if session._pending else {}
    draft_status = ("SUBMISSION_PENDING" if pending.get("accepted") is True else
                    "WAITING_CONFIRMATION" if pending else "NONE")
    result = session.conversation_state.get("last_tool_result") or {}
    last_order = session.last_order
    return {
        "draft_status": draft_status,
        "last_submitted_order_number": last_order,
        "last_verified_order_result": copy.deepcopy(result) if last_order and result.get("number") == last_order else None,
        "external_contact_supported": False,
        "external_contact_performed": False,
        "real_dispatch_supported": False,
        "completion_time_verified": False,
        "business_environment": "MOCK",
    }


def operation_answer(snapshot, subject):
    """Controlled facts, not a replacement for the model's contextual understanding."""
    contact = "我没有联系外部维修单位；这里的模拟报修不会自动联系真实维修人员。"
    status = snapshot["draft_status"]
    if status == "SUBMISSION_PENDING":
        receipt = "这次模拟报修的提交结果还没有核实，暂时不能确认新工单编号或维修进度。"
    elif status == "WAITING_CONFIRMATION":
        receipt = "当前只有待确认的报修草稿，尚未提交，因此还没有新工单编号或已核实的维修进度。"
    elif snapshot["last_submitted_order_number"]:
        receipt = "本会话最近提交的模拟工单编号是：" + snapshot["last_submitted_order_number"] + "。实际维修进度仍需查询该工单，不能据此认定已经联系维修单位或能在今天修好。"
    else:
        receipt = "这次故障尚未通过本会话提交报修，没有新工单编号，也没有已核实的维修进度。"
    return contact if subject == "EXTERNAL_CONTACT" else receipt if subject == "ORDER_RECEIPT" else contact + "\n" + receipt
