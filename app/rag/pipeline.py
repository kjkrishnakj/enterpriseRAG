from app.embeddings.embedder import embed
from app.core.config import RAGConfig
from app.rag.reranker import rerank

def answer_query(query, db, llm):
    query_embedding = embed([query])
    results = db.search(query_embedding, RAGConfig.TOP_K)

    if not results:
        return "I don't know"

    # Distance filtering
    filtered = [
        text for dist, text in results
        if dist < RAGConfig.SIMILARITY_THRESHOLD
    ]

    if not filtered:
        filtered = [results[0][1]]

    # 🔥 Re-ranking step
    reranked = rerank(query, filtered)

    if not reranked:
        return "I don't know"

    context = "\n".join(reranked[:2])  # only best chunks
    return llm(context, query)
