# build_embeddings.py
import numpy as np
from sentence_transformers import SentenceTransformer
import os

BASE_PATH = "./CORA/"

# =========================
# 1. LOAD PAPERS
# =========================
papers = {}
with open(os.path.join(BASE_PATH, "papers_dataset.txt"), "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(";")
        if len(parts) < 3:
            continue
        paper_id = parts[0]
        features = parts[2]
        papers[paper_id] = features

# =========================
# 2. LOAD WORDS DICTIONARY
# =========================
words_dict = {}
with open(os.path.join(BASE_PATH, "words_dictionary.txt"), "r", encoding="utf-8") as f:
    for line in f:
        word, code = line.strip().split()
        words_dict[code] = word

# =========================
# 3. DECODE FEATURES
# =========================
def decode_features(feature_string):
    words = []
    for item in feature_string.split(","):
        code = item.split(":")[0]
        if code in words_dict:
            words.append(words_dict[code])
    return " ".join(words)

# =========================
# 4. COMPUTE EMBEDDINGS
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")
paper_texts = {pid: decode_features(feat) for pid, feat in papers.items()}

paper_embeddings = {}
for pid, text in paper_texts.items():
    paper_embeddings[pid] = model.encode(text, normalize_embeddings=True)

# =========================
# 5. SAVE EMBEDDINGS
# =========================
np.save("paper_embeddings.npy", np.stack(list(paper_embeddings.values())))
np.save("paper_ids.npy", np.array(list(paper_embeddings.keys())))

print("Embeddings saved: paper_embeddings.npy, paper_ids.npy")