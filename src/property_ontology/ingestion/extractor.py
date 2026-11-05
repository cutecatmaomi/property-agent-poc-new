import re

from ..models import ExtractedClaim, digest, make_id

KEYWORDS = {
    "fee": ["物业费", "物业服务费", "收费标准", "费用", "元/", "元／", "保证金"],
    "decoration": ["装修", "施工时间", "噪音施工", "装饰"],
    "repair": ["报修", "保修", "质保", "日常维修", "入户维修"],
    "service": ["物业服务", "服务中心", "工作时间", "服务时间", "管家", "邮件"],
}


def extract_candidates(estate, chunks, limit=7):
    """Conservative, extractive baseline: every statement is an unchanged evidence excerpt."""
    selected = []
    used = set()
    for round_index in range(2):
        for topic, words in KEYWORDS.items():
            scored = []
            for chunk in chunks:
                text = chunk["text"]
                if not 30 <= len(text) <= 5000:
                    continue
                if (topic, chunk["id"]) in used:
                    continue
                score = sum(min(text.count(word), 4) for word in words)
                if score == 0:
                    continue
                filename = chunk["source_path"]
                score += 8 if any(word in filename for word in ["答客问", "百问百答"]) else 0
                score -= 8 if "销售" in filename else 0
                if round_index and chunk["id"] in {p["chunk_id"] for p in selected}:
                    continue
                scored.append((score, chunk))
            if not scored:
                continue
            _, chunk = max(scored, key=lambda value: value[0])
            lines = [line.strip() for line in chunk["text"].splitlines() if line.strip()]
            title = next((line[:100] for line in lines if any(word in line for word in words)), topic)
            claim = ExtractedClaim(
                topic=topic,
                title=title,
                statement=chunk["text"],
                conditions="保留原文中的业态、时间、暂定措辞和其他适用条件；需人工确认",
            )
            data = {
                **claim.model_dump(),
                "estate_id": estate,
                "chunk_id": chunk["id"],
                "document_id": chunk["document_id"],
                "extractor_version": "extractive-v1",
            }
            data["version_hash"] = digest(data)
            data["id"] = make_id(estate, "POL", data["version_hash"])
            data["status"] = "PENDING_REVIEW"
            selected.append(data)
            used.add((topic, chunk["id"]))
            if len(selected) >= limit:
                return selected
    return selected


def topic_for_question(question):
    counts = {
        topic: sum(len(re.findall(re.escape(word), question)) for word in words)
        for topic, words in KEYWORDS.items()
    }
    # A fee/amount question about decoration is primarily a fee lookup.
    if any(x in question for x in ("多少钱", "收费", "物业费", "押金", "保证金")):
        return "fee"
    return max(counts, key=counts.get) if any(counts.values()) else "service"
