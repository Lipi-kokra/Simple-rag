from pathlib import Path

from pypdf import PdfReader
from docx import Document

SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def _read_text(path: Path) -> str:
    return path.read_text(errors="ignore")


def load_documents(docs_dir: Path) -> list[tuple[str, str]]:
    """Return list of (source_filename, full_text) for every supported file in docs_dir."""
    out = []
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED:
            continue
        if path.suffix.lower() == ".pdf":
            text = _read_pdf(path)
        elif path.suffix.lower() == ".docx":
            text = _read_docx(path)
        else:
            text = _read_text(path)
        if text.strip():
            out.append((path.name, text))
    return out
