from fastapi import APIRouter, UploadFile, File
from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_text
from app.embeddings.embedder import embed
from app.vectorstore.faiss_db import FAISSDB
from app.rag.pipeline import answer_query
from app.models.llm import generate

router = APIRouter()
db = FAISSDB(dim=384)

@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    path = f"data/raw/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())

    text = load_document(path)
    chunks = chunk_text(text)
    embeddings = embed(chunks)
    db.add(embeddings, chunks)

    return {"chunks_indexed": len(chunks)}

@router.post("/ask")
def ask(question: str):
    answer = answer_query(question, db, generate)
    return {"answer": answer}
