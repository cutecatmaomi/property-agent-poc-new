import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path


def render_docx(path: Path, output: Path, soffice_path=""):
    executable = soffice_path or shutil.which("soffice")
    if not executable:
        raise RuntimeError("缺少LibreOffice；扫描DOCX需要渲染后OCR，请配置SOFFICE_PATH")
    output.mkdir(parents=True, exist_ok=True)
    # A private profile prevents collisions with an interactive office process.
    with tempfile.TemporaryDirectory(prefix="property-poc-office-") as profile:
        process = subprocess.run(
            [
                executable,
                f"-env:UserInstallation={Path(profile).as_uri()}",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(output.resolve()),
                str(path.resolve()),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
    target = output / (path.stem + ".pdf")
    if process.returncode or not target.is_file():
        raise RuntimeError("DOCX渲染失败，原件保留并等待人工核验")
    return target


def ocr_pdf(path: Path, output: Path, *, source_kind="pdf", engine=None):
    from importlib.metadata import version

    import pypdfium2 as pdfium

    if engine is None:
        from rapidocr import RapidOCR

        engine = RapidOCR()
    output.mkdir(parents=True, exist_ok=True)
    rendered_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    pdf = pdfium.PdfDocument(str(path))
    try:
        for index in range(len(pdf)):
            page = pdf[index]
            bitmap = page.render(scale=3)
            pil = bitmap.to_pil()
            image_path = output / f"page-{index + 1:03}.png"
            pil.save(image_path)
            result = engine(str(image_path))
            texts = list(result.txts) if result.txts is not None else []
            scores = [float(s) for s in result.scores] if result.scores is not None else []
            boxes = result.boxes.tolist() if result.boxes is not None else []
            yield {
                "text": "\n".join(texts),
                "raw_ocr_text": "\n".join(texts),
                "locator": {
                    "kind": "docx_rendered_page" if source_kind == "docx" else "ocr_pdf_page",
                    "page": index + 1,
                    "page_label": "渲染页" if source_kind == "docx" else "PDF物理页",
                    "render_sha256": rendered_hash,
                    "image_path": str(image_path),
                    "boxes": boxes,
                },
                "ocr_scores": scores,
                "ocr_engine": "RapidOCR",
                "ocr_version": version("rapidocr"),
                "extraction_method": "ocr",
                "quality": "NEEDS_HUMAN_VERIFICATION",
            }
            pil.close()
            bitmap.close()
            page.close()
    finally:
        pdf.close()
