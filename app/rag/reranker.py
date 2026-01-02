import re

STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "of", "to", "in", "on",
    "for", "and", "or", "with", "how", "many", "much"
}

def rerank(query: str, chunks: list[str]) -> list[str]:
    query_words = {
        w.lower() for w in re.findall(r"\w+", query)
        if w.lower() not in STOPWORDS
    }

    scored = []

    for chunk in chunks:
        chunk_words = set(re.findall(r"\w+", chunk.lower()))
        score = len(query_words & chunk_words)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [chunk for score, chunk in scored if score > 0]
