import faiss
import pickle
import os

class FAISSDB:
    def __init__(self, dim, index_path="data/index/faiss.index", meta_path="data/index/texts.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.texts = None

    def _load(self):
        if self.index is not None:
            return

        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.texts = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dim)
            self.texts = []

    def add(self, embeddings, texts):
        self._load()
        self.index.add(embeddings)
        self.texts.extend(texts)
        self._save()

    def search(self, query_embedding, k=3):
        self._load()
        D, I = self.index.search(query_embedding, k)
        return [(D[0][i], self.texts[idx]) for i, idx in enumerate(I[0]) if idx != -1]

    def _save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.texts, f)
