from app.core.config import RAGConfig

def chunk_text(text: str):
    size = RAGConfig.CHUNK_SIZE
    overlap = RAGConfig.CHUNK_OVERLAP

    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        chunk = words[i:i + size]
        chunks.append(" ".join(chunk))
        i += size - overlap

    return chunks
