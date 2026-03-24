import fitz  # PyMuPDF
import re


def load_and_chunk_pdf(file_path: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    Extract text from a PDF and split it into overlapping chunks.

    Args:
        file_path: Path to the PDF file.
        chunk_size: Number of words per chunk.
        overlap: Number of words to overlap between chunks.

    Returns:
        List of text chunks.
    """
    doc = fitz.open(file_path)
    full_text = ""

    for page in doc:
        full_text += page.get_text()

    doc.close()

    # Clean up whitespace
    full_text = re.sub(r'\s+', ' ', full_text).strip()

    if not full_text:
        return []

    # Split into overlapping word-based chunks
    words = full_text.split()
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks