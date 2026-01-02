from pypdf import PdfReader
from pathlib import Path

def load_document(path: str) -> str:
    ext = Path(path).suffix.lower()

    if ext == ".pdf":
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    raise ValueError("Unsupported file type")
