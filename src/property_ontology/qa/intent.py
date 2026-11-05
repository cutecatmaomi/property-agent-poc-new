"""Shared concept/facet matching, not per-question overrides. No corpus answers are hard-coded."""

import re
import unicodedata
from dataclasses import dataclass

from ..response_policy import normalized_text

SUBJECTS = {
    "property_service": ("物业服务", "物业工作"),
    "property_fee": ("物业费", "物业服务费", "物管费"),
    "property_center": ("物业服务中心", "物业中心", "客户服务前台", "客服前台"),
    "maintenance_center": ("维保服务中心", "维保中心", "维修中心"),
    "butler": ("楼栋管家", "物业管家", "管家"),
    "control_room": ("中控室", "监控室"),
    "repair": ("报修", "维保", "维修", "整改"),
    "renovation": ("装修", "施工", "装饰工程"),
    "deposit": ("押金", "保证金"),
    "parking": ("停车", "临停", "车位", "车库", "充电桩"),
    "warranty": ("保修", "质保", "保质期"),
    "roof": ("屋面", "屋顶"), "bathroom": ("卫生间", "洗手间"),
    "external_wall": ("外墙",), "wall": ("墙面",), "floor": ("地面",),
    "door_window": ("门窗", "窗户", "窗", "木门"), "structure": ("主体结构", "地基", "基础设施"),
    "heating": ("供热", "采暖",), "cooling": ("供冷",), "electrical": ("电气管线",),
    "water_piping": ("给排水管",), "equipment": ("设备安装",),
    "access_card": ("门禁卡",), "garbage": ("垃圾", "清运费"),
    "worker_pass": ("施工人员出入证", "施工出入证"), "extinguisher": ("灭火器",),
    "mail": ("邮件", "代收",), "contract": ("合同",), "invoice": ("发票",),
    "delivery": ("交付", "收房",), "complaint": ("投诉",),
    "activity": ("活动", "报名",), "membership": ("积分", "会员",),
    "city": ("公交", "医院", "学校", "地铁",), "project": ("项目", "楼盘",),
}
FACETS = {
    "location": ("哪里", "在哪", "位置", "地址", "何处"),
    "phone": ("电话", "联系号码", "联系方式", "如何联系", "怎么联系"),
    "hours": ("几点", "时段", "上班", "下班", "工作时间", "服务时间", "施工时间", "什么时间"),
    "price": ("多少钱", "多少", "费用", "收费", "标准", "价格", "费率"),
    "duration": ("多久", "期限", "几年", "年限", "多长时间", "维修时限", "整改时间"),
    "refund": ("退还", "退回", "退押金", "退款"),
    "scope": ("范围", "包括", "哪些服务", "负责什么", "职责"),
    "channels": ("渠道", "途径", "怎么报修", "如何报修"),
    "procedure": ("流程", "手续", "怎么办", "如何办理", "怎么处理", "怎么拿", "领取"),
    "components": ("组成", "构成", "明细", "包含哪些费用"),
}
SERVICE_SUBJECTS = {
    "property_fee": {"property_fee"}, "renovation": {"renovation", "renovation_deposit", "garbage", "worker_pass", "extinguisher"},
    "maintenance": {"repair", "maintenance_center"}, "property_service": {"property_service", "property_center", "butler", "control_room", "mail", "access_card"},
    "warranty": {"warranty"}, "parking": {"parking"}, "complaint": {"complaint"},
    "purchase": {"contract", "invoice"}, "delivery": {"delivery"}, "activity": {"activity"},
    "membership": {"membership"}, "city_info": {"city"}, "project_info": {"project"},
}
SPECIFIC = {"property_center", "maintenance_center", "butler", "control_room", "property_fee", "renovation_deposit",
            "roof", "bathroom", "external_wall", "wall", "floor", "door_window", "structure", "heating", "cooling",
            "electrical", "water_piping", "equipment", "access_card", "garbage", "worker_pass", "extinguisher", "mail", "contract", "invoice"}
PARTS = SPECIFIC & {"roof", "bathroom", "external_wall", "wall", "floor", "door_window", "structure", "heating", "cooling", "electrical", "water_piping", "equipment"}


def canonical_question(value):
    text = normalized_text(value)
    for words in SUBJECTS.values():
        for alias in sorted(words, key=len, reverse=True):
            text = text.replace(normalized_text(alias), normalized_text(words[0]))
    return re.sub(r"请问|您好|你好|我想知道|能否告诉我|可以告诉我|一下", "", text)


def subjects(text):
    result = {k for k, words in SUBJECTS.items() if any(w in text for w in words)}
    if "renovation" in result and "deposit" in result and "extinguisher" not in result:
        result.add("renovation_deposit")
        result -= {"renovation", "deposit"}
    if "maintenance_center" in result:
        result.discard("repair")
    return frozenset(result)


def facets(text, *, topic=None, fact=None):
    result = {k for k, words in FACETS.items() if any(w in text for w in words)}
    if result & {"phone", "hours", "duration"} and not re.search(r"费|金额|价格|保证金|押金", text):
        result.discard("price")
    if topic == "fee":
        result.add("price")
    if topic == "hours":
        result.add("hours")
    if topic == "warranty":
        result.add("duration")
    fact = fact or {}
    predicate, key = fact.get("predicate", ""), fact.get("fact_key", "")
    if predicate in {"location", "posted_at"}:
        result.add("location")
    if predicate == "channel":
        result.add("channels")
    if ".hours." in key:
        result.add("hours")
    if "phone" in key or "contact" in key and re.search(r"电话|热线", text):
        result.add("phone")
    if ".duty." in key:
        result.add("scope")
    if predicate in {"duration", "estimated_duration"} or "estimate" in key:
        result.add("duration")
    if "refund" in key:
        result.add("refund")
    if fact.get("value_role") in {"TOTAL", "COMPONENT"}:
        result.add("components")
    if fact and topic != "fee":
        result.discard("price")
    return frozenset(result)


def grams(value):
    text = canonical_question(value)
    return {text[i:i + 2] for i in range(len(text) - 1)}


@dataclass(frozen=True)
class QuestionScope:
    question: str
    estate_id: str | None
    service_code: str | None
    subjects: frozenset[str]
    facets: frozenset[str]
    inferred_service_code: str | None = None

    @classmethod
    def parse(cls, question, estate_id, service_code=None):
        s, f = subjects(question), facets(question)
        if "property_fee" in s and not f:
            f = frozenset({"price"})
        if not f and re.search(r"费|金额|价格|押金|保证金", question):
            f = frozenset({"price"})
        inferred = {code for code, names in SERVICE_SUBJECTS.items() if names & s}
        if "warranty" in s:
            inferred = {"warranty"}
        code = next(iter(inferred)) if len(inferred) == 1 else None
        return cls(question, estate_id, service_code, s, f, code)

    def score(self, title, question, service_code, topic=None, fact=None, entity_context=""):
        if self.service_code and self.service_code != service_code:
            return 0.0
        description = title + " " + question
        cs, cf = subjects(description + " " + entity_context), facets(description, topic=topic, fact=fact)
        if service_code == "property_service":
            cs = cs | {"property_service"}
        if service_code == "property_fee":
            cs = cs | {"property_fee"}
        qspecific, cspecific = self.subjects & SPECIFIC, cs & SPECIFIC
        if qspecific and not qspecific.intersection(cspecific):
            return 0.0
        if canonical_question(self.question) in {canonical_question(question), canonical_question(title)}:
            return 1.0
        if self.facets and cf and not any(facet_matches(f, cf) for f in self.facets):
            return 0.0
        qg, cg = grams(self.question), grams(description)
        overlap = len(qg & cg) / max(1, len(qg))
        exact_entity = bool(qspecific & cspecific)
        family_match = bool(self.subjects & cs)
        if not qspecific and self.subjects and (self.service_code or self.inferred_service_code) == service_code:
            family_match = True
        if not family_match and overlap < 0.5:
            return 0.0
        if self.facets and not cf:
            return 0.0
        if self.subjects and not qspecific:
            services = {code for code, names in SERVICE_SUBJECTS.items() if names & self.subjects}
            if "warranty" in self.subjects:
                services = {"warranty"}
            if services and service_code not in services and not exact_entity:
                return 0.0
        return min(0.99, (0.64 if exact_entity else 0.6 if family_match and not self.facets else 0.46 if family_match else 0.0)
                   + 0.18 * any(facet_matches(f, cf) for f in self.facets) + 0.18 * overlap)


def facet_matches(requested, offered):
    return bool(({"channels", "phone"} if requested == "channels" else {requested}) & set(offered))


def normalized_answer(text):
    """Only formatting normalization; digits, decimal points, time separators and words remain intact."""
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s*\n\s*(?=[\u4e00-\u9fff])", "", text)
    return re.sub(r"\s+", " ", text).strip()


def content_key(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).strip("。.!！")
