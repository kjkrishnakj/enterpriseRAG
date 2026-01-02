from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Enterprise Knowledge Base RAG")

app.include_router(router)

@app.get("/")
def health():
    return {"status": "running"}
