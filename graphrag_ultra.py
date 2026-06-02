import numpy as np
import faiss
import ollama
from sentence_transformers import SentenceTransformer
from collections import defaultdict

BASE_PATH = "./CORA/"
TOP_K = 10
HOPS = 1
DAMPING = 0.85
ITER = 20

# =========================
# LOAD PAPERS
# =========================
papers = {}
with open(BASE_PATH + "papers_dataset.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(";")
        if len(parts) < 3:
            continue
        papers[parts[0]] = parts[2]

# =========================
# WORD DICT
# =========================
words_dict = {}
with open(BASE_PATH + "words_dictionary.txt", "r", encoding="utf-8") as f:
    for line in f:
        w, c = line.strip().split()
        words_dict[c] = w

def decode(x):
    out = []
    for t in x.split(","):
        c = t.split(":")[0]
        if c in words_dict:
            out.append(words_dict[c])
    return " ".join(out)

# =========================
# CITATION GRAPH
# =========================
graph = defaultdict(list)
reverse = defaultdict(list)

with open(BASE_PATH + "citations.txt", "r") as f:
    for line in f:
        inside = line.split("(")[1].split(")")[0]
        s, d = inside.split(",")
        graph[s].append(d)
        reverse[d].append(s)

nodes = list(papers.keys())

# =========================
# PAGE RANK (REAL)
# =========================
def pagerank():
    N = len(nodes)
    rank = {n: 1/N for n in nodes}

    for _ in range(ITER):
        new_rank = {n: (1 - DAMPING) / N for n in nodes}

        for n in nodes:
            for out in graph[n]:
                if len(graph[n]) > 0:
                    new_rank[out] += DAMPING * rank[n] / len(graph[n])

        rank = new_rank

    return rank

pagerank_score = pagerank()

# =========================
# FAISS
# =========================
embeddings = np.load("paper_embeddings.npy")
paper_ids = np.load("paper_ids.npy")

model = SentenceTransformer("all-MiniLM-L6-v2")

faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# =========================
# HYBRID RETRIEVAL + RERANK
# =========================
def retrieve(query):
    q = model.encode(query, normalize_embeddings=True)

    D, I = index.search(q.reshape(1, -1), TOP_K)

    candidates = []
    for score, idx in zip(D[0], I[0]):
        pid = paper_ids[idx]
        pr = pagerank_score.get(pid, 0)

        final = 0.6 * score + 0.4 * pr
        candidates.append((final, pid))

    candidates.sort(reverse=True)
    return [p for _, p in candidates[:5]]

# =========================
# GRAPH EXPANSION
# =========================
def expand(seed):
    visited = set(seed)
    frontier = set(seed)

    for _ in range(HOPS):
        nxt = set()
        for n in frontier:
            nxt.update(graph[n])
        visited.update(nxt)
        frontier = nxt

    return visited

# =========================
# CONTEXT COMPRESSION
# =========================
def build_context(seeds):
    nodes = expand(seeds)

    texts = []
    for pid in nodes:
        if pid in papers:
            txt = decode(papers[pid])
            texts.append(f"{pid}: {txt}")

    # LIMIT CONTEXT (VERY IMPORTANT)
    return "\n".join(texts[:20])

# =========================
# ASK
# =========================
def ask(q):
    seeds = retrieve(q)
    context = build_context(seeds)

    prompt = f"""
You are a scientific assistant.

Use ONLY the context below.

Summarize and explain relationships between papers.

QUESTION:
{q}

CONTEXT:
{context}

Answer clearly and avoid repetition.
"""

    res = ollama.chat(
        model="gemma3:4b",
        messages=[{"role": "user", "content": prompt}]
    )

    return res["message"]["content"]

# =========================
# CLI
# =========================
if __name__ == "__main__":
    print("🔥 Ultra GraphRAG ready (exit to quit)\n")

    while True:
        q = input("Question: ")
        if q == "exit":
            break

        print("\nAnswer:\n")
        print(ask(q))
        print("\n" + "-"*60 + "\n")