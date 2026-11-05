"""Day 1 offline commands; intentionally contains no database or agent entry point."""

import argparse
import hashlib
import json
from dataclasses import replace

from .config import Settings
from .ingestion.docx import inspect_docx
from .ingestion.pipeline import extract_corpus
from .reports import build_day1_reports


def main():
    parser = argparse.ArgumentParser(description="Day 1资料检查、离线提取与Ontology Schema")
    parser.add_argument("command", choices=["inspect", "extract", "report"])
    parser.add_argument("--ocr", action="store_true", help="对扫描文档渲染后执行本地OCR")
    parser.add_argument("--estate", choices=["E001", "E002"])
    parser.add_argument("--soffice", help="LibreOffice soffice完整路径")
    args = parser.parse_args()
    settings = Settings.load()
    if args.soffice:
        settings = replace(settings, soffice_path=args.soffice)
    if args.command == "inspect":
        rows = []
        for estate, config in settings.estates().items():
            for path in sorted((settings.root / "data/raw" / config["raw_directory"]).iterdir()):
                if path.suffix.lower() not in {".pdf", ".docx"}:
                    continue
                rows.append(
                    {
                        "estate_id": estate,
                        "file": str(path.relative_to(settings.root)),
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        **(inspect_docx(path) if path.suffix.lower() == ".docx" else {}),
                    }
                )
        result = rows
    elif args.command == "extract":
        result = extract_corpus(settings, enable_ocr=args.ocr, only=args.estate)
        build_day1_reports(settings)
    else:
        result = build_day1_reports(settings)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
