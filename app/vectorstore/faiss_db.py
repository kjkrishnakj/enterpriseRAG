import faiss
import pickle
import os

class FAISSDB:
    def __init__(self, dim, index_path="data/index/faiss.index", meta_path="data/index/texts.pkl"):
        self.index_path = index_path
        self.meta_path = meta_path
        self.dim = dim

        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                self.texts = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.texts = []

    def add(self, embeddings, texts):
        self.index.add(embeddings)
        self.texts.extend(texts)
        self._save()

    def search(self, query_embedding, k=3):
        D, I = self.index.search(query_embedding, k)
        results = []

        for dist, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            results.append((dist, self.texts[idx]))

        return results

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.texts, f)
