# build_index.py
import faiss
import numpy as np

# =========================
# 1. LOAD EMBEDDINGS
# =========================
embeddings = np.load("paper_embeddings.npy")
paper_ids = np.load("paper_ids.npy")

# =========================
# 2. BUILD FAISS INDEX
# =========================
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # Inner Product for cosine similarity
faiss.normalize_L2(embeddings)  # normalize vectors for cosine

index.add(embeddings)
print(f"FAISS index built with {index.ntotal} papers.")

# =========================
# 3. SEARCH FUNCTION
# =========================
def search(query_vec, k=5):
    faiss.normalize_L2(query_vec)
    D, I = index.search(query_vec.reshape(1, -1), k)
    return [paper_ids[i] for i in I[0]]

# =========================
# 4. TEST SEARCH
# =========================
query_vec = embeddings[0].reshape(1, -1)
results = search(query_vec, k=5)
print("Top papers for first vector:", results)