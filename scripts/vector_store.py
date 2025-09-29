def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimpleVectorStore:
    def __init__(self, texts):
        self.texts = texts
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.model.encode(texts)

    def similarity_search(self, query, k=4):
        query_vec = self.model.encode([query])
        sims = cosine_similarity(query_vec, self.embeddings)[0]
        top_k_idx = np.argsort(sims)[::-1][:k]
        return [self.texts[i] for i in top_k_idx] 