from pathlib import Path

import pdfplumber


def parse_pdf(path: Path):
    with pdfplumber.open(path) as pdf:
        for index, page in enumerate(pdf.pages, 1):
            text = (page.extract_text() or "").strip()
            if text:
                yield {
                    "text": text,
                    "locator": {"kind": "pdf_page", "page": index, "bbox": list(page.bbox)},
                    "extraction_method": "native_pdf",
                    "quality": "RELIABLE" if len(text) >= 30 else "NEEDS_HUMAN_VERIFICATION",
                }
            for table_index, table in enumerate(page.find_tables(), 1):
                rows = table.extract()
                for row_index, row in enumerate(rows, 1):
                    values = [str(cell or "").replace("\n", " ") for cell in row]
                    if any(values):
                        yield {
                            "text": " | ".join(values),
                            "locator": {
                                "kind": "pdf_table",
                                "page": index,
                                "table": table_index,
                                "row": row_index,
                                "bbox": list(table.bbox),
                            },
                            "extraction_method": "native_pdf_table",
                            "quality": "RELIABLE",
                        }
            if not text:
                yield {
                    "text": "",
                    "locator": {"kind": "pdf_page", "page": index},
                    "extraction_method": "ocr_required",
                    "quality": "NEEDS_HUMAN_VERIFICATION",
                }
