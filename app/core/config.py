import os

class RAGConfig:
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

    TOP_K = int(os.getenv("TOP_K", 3))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 1.5))

    MODEL_NAME = os.getenv("MODEL_NAME", "google/flan-t5-base")
    MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 64))
