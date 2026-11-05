import hashlib
import json

from ..models import digest, make_id, now_iso
from .docx import inspect_docx, parse_docx
from .extractor import extract_candidates
from .ocr import ocr_pdf, render_docx
from .pdf import parse_pdf


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def stable_chunk_id(estate, document_id, locator, text, parser_version="native-v1", ocr_version=None):
    # Render metadata contains timestamps; it must not change the identity of identical source evidence.
    identity_locator = {k: v for k, v in locator.items() if k not in {"render_sha256", "image_path"}}
    return make_id(
        estate, "CHUNK", [document_id, identity_locator, digest(text), parser_version, ocr_version]
    )


def extract_corpus(settings, *, enable_ocr=False, candidate_limit=7, only=None):
    summaries = []
    for estate, config in settings.estates().items():
        if only and estate != only:
            continue
        output = settings.root / "data/derived" / estate
        output.mkdir(parents=True, exist_ok=True)
        documents, all_chunks = [], []
        raw = settings.root / "data/raw" / config["raw_directory"]
        files = sorted(p for p in raw.iterdir() if p.suffix.lower() in {".docx", ".pdf"})
        for path in files:
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            relative = path.relative_to(settings.root).as_posix()
            document_id = make_id(estate, "DOC", [relative, sha])
            document = {
                "id": document_id,
                "estate_id": estate,
                "source_path": relative,
                "sha256": sha,
                "filename": path.name,
                "last_parsed_at": now_iso(),
                "verification_status": "PENDING_VERIFICATION",
                "parser_version": "native-v1",
            }
            issues = ["VERSION_AND_APPLICABILITY_REQUIRE_HUMAN_REVIEW"]
            extracted = []
            try:
                if path.suffix.lower() == ".docx":
                    metadata = inspect_docx(path)
                    document.update(metadata)
                    if metadata["requires_ocr"]:
                        if enable_ocr:
                            rendered = render_docx(path, output / "renders" / sha, settings.soffice_path)
                            extracted = list(ocr_pdf(rendered, output / "ocr" / sha, source_kind="docx"))
                            document["parse_status"] = "OCR_NEEDS_HUMAN_VERIFICATION"
                        else:
                            document["parse_status"] = "OCR_REQUIRED"
                            issues.append("NO_NATIVE_TEXT_FLOATING_IMAGES")
                    else:
                        extracted = list(parse_docx(path))
                        document["parse_status"] = "TEXT_EXTRACTED"
                        if metadata["image_placements"]:
                            document["parse_status"] = "TEXT_EXTRACTED_IMAGES_PENDING_REVIEW"
                            issues.append("EMBEDDED_IMAGES_NOT_OCR_REVIEW_REQUIRED")
                else:
                    extracted = list(parse_pdf(path))
                    if any(c["extraction_method"] == "ocr_required" for c in extracted):
                        if enable_ocr:
                            extracted = list(ocr_pdf(path, output / "ocr" / sha))
                            document["parse_status"] = "OCR_NEEDS_HUMAN_VERIFICATION"
                        else:
                            document["parse_status"] = "PARTIAL_OCR_REQUIRED"
                    else:
                        document["parse_status"] = "TEXT_EXTRACTED"
                fulltext = "\n".join(chunk["text"] for chunk in extracted)
                native_chars = document.get("native_text_chars", 0)
                if native_chars and len(fulltext) < native_chars * 0.90:
                    document["parse_status"] = "PARTIAL_NATIVE_TEXT_EXTRACTION"
                    issues.append("TEXTBOX_OR_LAYOUT_TEXT_REQUIRES_MANUAL_CHECK")
                if estate == "E001" and any(name in fulltext for name in ["双城汇", "华润中心"]):
                    issues.append("ESTATE_NAME_MISMATCH")
                if "客户标准化手册" in fulltext or "济南中心" in fulltext:
                    issues.append("TEMPLATE_HEADER_APPLICABILITY_REQUIRES_REVIEW")
                document["text_chars"] = len(fulltext)
            except Exception as error:
                document["parse_status"] = "FAILED_NEEDS_HUMAN_VERIFICATION"
                document["error"] = f"{type(error).__name__}: {str(error)[:250]}"
                issues.append("EXTRACTION_FAILED_NO_CONTENT_INVENTED")
            if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                raise RuntimeError("原始资料哈希发生变化，停止导入")
            document["issues_json"] = json.dumps(issues, ensure_ascii=False)
            documents.append(document)
            for item in extracted:
                chunk = {
                    **item,
                    "estate_id": estate,
                    "document_id": document_id,
                    "source_path": relative,
                    "source_sha256": sha,
                    "parser_version": "native-v1",
                }
                locator = chunk.pop("locator")
                chunk["locator_json"] = json.dumps(locator, ensure_ascii=False)
                chunk["content_hash"] = digest(chunk["text"])
                chunk["id"] = stable_chunk_id(
                    estate,
                    document_id,
                    locator,
                    chunk["text"],
                    chunk["parser_version"],
                    chunk.get("ocr_version"),
                )
                if "ocr_scores" in chunk:
                    chunk["ocr_scores_json"] = json.dumps(chunk.pop("ocr_scores"))
                all_chunks.append(chunk)
        candidates = extract_candidates(estate, all_chunks, candidate_limit)
        write_jsonl(output / "documents.jsonl", documents)
        write_jsonl(output / "evidence.jsonl", all_chunks)
        write_jsonl(output / "candidates.jsonl", candidates)
        summary = {
            "estate_id": estate,
            "documents": len(documents),
            "evidence_chunks": len(all_chunks),
            "candidates": len(candidates),
            "approved_by_pipeline": 0,
            "statuses": {d["filename"]: d["parse_status"] for d in documents},
        }
        (output / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        summaries.append(summary)
    return summaries
