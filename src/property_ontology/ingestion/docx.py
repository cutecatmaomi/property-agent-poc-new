from pathlib import Path
from zipfile import ZipFile

from docx import Document
from docx.text.paragraph import Paragraph
from lxml import etree

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}


def inspect_docx(path: Path):
    with ZipFile(path) as archive:
        xml = etree.fromstring(archive.read("word/document.xml"))
        text = "".join(xml.xpath("//w:t/text()", namespaces=NS))
        images = len(xml.xpath("//a:blip", namespaces=NS))
        return {
            "native_text_chars": len(text.strip()),
            "image_placements": images,
            "floating_images": len(xml.xpath("//*[local-name()='anchor']")),
            "requires_ocr": len(text.strip()) < 100 and images > 0,
        }


def parse_docx(path: Path):
    """Preserve XML paragraph and table locators, never invent Word page numbers."""
    document = Document(path)
    pending = []
    length = 0

    def flush():
        nonlocal pending, length
        if not pending:
            return None
        item = {
            "text": "\n".join(t for t, _ in pending),
            "locator": {"kind": "docx_xml", "segments": [loc for _, loc in pending]},
            "extraction_method": "native_docx",
            "quality": "RELIABLE",
        }
        pending, length = [], 0
        return item

    for index, p in enumerate(document.element.body.iterchildren(), 1):
        if p.tag == f"{{{NS['w']}}}p":
            text = Paragraph(p, document).text.strip()
            if not text:
                continue
            if length > 900:
                yield flush()
            pending.append((text, {"body_child": index, "xml_path": p.getroottree().getpath(p)}))
            length += len(text)
        elif p.tag == f"{{{NS['w']}}}tbl":
            if item := flush():
                yield item
            for row_index, row in enumerate(p.findall("w:tr", NS), 1):
                cells = []
                for cell_index, cell in enumerate(row.findall("w:tc", NS), 1):
                    paragraphs = cell.findall(".//w:p", NS)
                    for para_index, para in enumerate(paragraphs, 1):
                        text = Paragraph(para, document).text.strip()
                        if text:
                            cells.append(
                                (
                                    text,
                                    {
                                        "body_child": index,
                                        "row": row_index,
                                        "column": cell_index,
                                        "paragraph": para_index,
                                        "xml_path": para.getroottree().getpath(para),
                                    },
                                )
                            )
                # PDF-converted Word files may put an entire page in one table cell.
                for text, locator in cells:
                    if length > 900:
                        yield flush()
                    pending.append((text, locator))
                    length += len(text)
                if item := flush():
                    yield item
    if item := flush():
        yield item
