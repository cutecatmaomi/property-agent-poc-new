"""Deterministic Day 1/2 response routing; no model, database or external contact."""

import re
import unicodedata

FIXED_SERVICE_REPLY = (
    "我是华润置地客服小助手，只能提供报事报修、活动查询和报名、城市资讯查询以及项目基础信息查询等相关功能！"
    "如果您有以上相关服务类问题，欢迎您再次联系我；如果您需要人工服务，可以拨打华润置地客户服务热线"
    "4001099888，我们将竭诚为您服务"
)

# A local POC topic detector, not a clinical assessment or a claim of complete semantic coverage.
# Normalize spaces/punctuation/invisible characters so trivial spelling changes do not skip routing.
RED_LINE_TERMS = (
    "自杀", "自殺", "跳楼", "跳樓", "轻生", "輕生", "寻短见", "尋短見",
    "不想活", "活不下去", "不想再活", "不想继续活", "结束生命", "結束生命",
    "结束自己的生命", "結束自己的生命", "结束我的生命", "想死", "去死", "割腕",
    "伤害自己", "傷害自己", "跳下去", "跳河", "跳海", "上吊",
    "suicide", "suicidal", "killmyself", "endmylife", "selfharm",
)
HUMAN_SERVICE_TERMS = (
    "转人工", "轉人工", "找人工", "人工客服", "人工服务", "人工服務",
    "真人客服", "人工处理", "人工處理", "人工核验", "人工核驗",
    "人工热线", "humanagent", "humanservice",
)


def normalized_text(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(c for c in text if unicodedata.category(c)[0] in {"L", "N"})


def classify_message(text):
    """Return a server-determined reason; red-line topics always have priority."""
    normalized = normalized_text(text)
    # An amusement ride is not itself a self-harm topic; other terms still take priority.
    risk_text = normalized.replace("跳楼机", "").replace("跳樓機", "")
    if any(term in risk_text for term in RED_LINE_TERMS) or re.search(
        r"(?:从|從).{0,12}(?:楼|樓|天台|阳台|陽台).{0,8}跳(?:下|出去)", risk_text
    ):
        return "RED_LINE"
    if any(term in normalized for term in HUMAN_SERVICE_TERMS):
        return "USER_REQUEST"
    return None


def pending_question_matches(question, candidate_question):
    """Conservative title matching; short generic fragments do not claim a relevant source exists."""
    query, title = normalized_text(question), normalized_text(candidate_question)
    return bool(query and title) and (query == title or min(len(query), len(title)) >= 4
                                     and (query in title or title in query))


def review_required_reply(estate_id, knowledge_mode, *, demo_content_confirmed=False):
    if demo_content_confirmed:
        message = ("该问答已有人工确认的POC演示记录；现行政策的适用范围与真实有效期尚未核验，暂不能作为现行政策答复。"
                   "查看已确认的资料原文请使用演示模式。" if knowledge_mode == "current" else
                   "相关资料已有人工内容确认记录，但来源、版本、适用条件或冲突核验尚未通过，暂不能提供可靠答复。")
        message += "如需人工服务，可以拨打华润置地客户服务热线4001099888。"
    else:
        message = ("已找到相关资料，但尚未完成人工审核、适用范围及有效期核验，暂不能作为已确认答复。"
                   "如需人工服务，可以拨打华润置地客户服务热线4001099888。")
    return {
        "estate_id": estate_id,
        "response_type": "REVIEW_REQUIRED",
        "reason_code": "KNOWLEDGE_PENDING_REVIEW",
        "knowledge_mode": "CURRENT_POLICY" if knowledge_mode == "current" else "POC_DEMO",
        "red_line": False,
        "message": message,
        "demo_content_confirmed": demo_content_confirmed,
        "matches": [],
        "review_status": "PENDING_REVIEW",
        "contact_channel": "PHONE",
        "contact_phone": "4001099888",
        "created": False,
    }


def fixed_reply(estate_id, reason_code):
    if reason_code not in {"RED_LINE", "USER_REQUEST", "NO_VERIFIED_MATCH"}:
        raise ValueError("Unsupported fixed reply reason")
    return {
        "estate_id": estate_id,
        "response_type": "FIXED_REPLY",
        "reason_code": reason_code,
        "red_line": reason_code == "RED_LINE",
        "message": FIXED_SERVICE_REPLY,
        "matches": [],
        "contact_channel": "PHONE",
        "contact_phone": "4001099888",
        "created": False,
    }
