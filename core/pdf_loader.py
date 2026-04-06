import io
import re

import fitz  # PyMuPDF


def _extract_text_from_document(doc: fitz.Document) -> str:
    full_text = ""
    for page in doc:
        page_text = page.get_text("text")
        if isinstance(page_text, str):
            full_text += page_text
        elif page_text is not None:
            full_text += str(page_text)

    return re.sub(r"\s+", " ", full_text).strip()


def _chunk_text(full_text: str, chunk_size: int, overlap: int) -> list[str]:
    if not full_text:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = full_text.split()
    chunks: list[str] = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def load_and_chunk_pdf(file_path: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Extract text from a PDF path and split it into overlapping chunks."""
    doc = fitz.open(file_path)
    try:
        full_text = _extract_text_from_document(doc)
    finally:
        doc.close()

    return _chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)


def load_and_chunk_pdf_bytes(pdf_bytes: bytes, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Extract text from PDF bytes and split it into overlapping chunks."""
    stream = io.BytesIO(pdf_bytes)
    doc = fitz.open(stream=stream, filetype="pdf")
    try:
        full_text = _extract_text_from_document(doc)
    finally:
        doc.close()

    return _chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
