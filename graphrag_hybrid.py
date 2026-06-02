import numpy as np
import faiss
import ollama
from sentence_transformers import SentenceTransformer
from collections import defaultdict

BASE_PATH = "./CORA/"
TOP_K = 8
HOPS = 1
ALPHA = 0.7  # FAISS weight
BETA = 0.3   # citation weight

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
# WORDS DICT
# =========================
words_dict = {}
with open(BASE_PATH + "words_dictionary.txt", "r", encoding="utf-8") as f:
    for line in f:
        w, c = line.strip().split()
        words_dict[c] = w

def decode(text):
    out = []
    for t in text.split(","):
        code = t.split(":")[0]
        if code in words_dict:
            out.append(words_dict[code])
    return " ".join(out)

# =========================
# CITATION GRAPH
# =========================
graph = defaultdict(list)
reverse_graph = defaultdict(list)

with open(BASE_PATH + "citations.txt", "r") as f:
    for line in f:
        inside = line.split("(")[1].split(")")[0]
        src, dst = inside.split(",")
        graph[src].append(dst)
        reverse_graph[dst].append(src)

# =========================
# SIMPLE CITATION SCORE (PageRank-like)
# =========================
def citation_score(pid):
    return len(reverse_graph[pid])  # indegree (simple importance)

# normalize citation scores
def normalize_scores(papers_list):
    scores = [citation_score(p) for p in papers_list]
    max_s = max(scores) if scores else 1
    return {p: citation_score(p)/max_s for p in papers_list}

# =========================
# LOAD FAISS
# =========================
embeddings = np.load("paper_embeddings.npy")
paper_ids = np.load("paper_ids.npy")

model = SentenceTransformer("all-MiniLM-L6-v2")

faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# =========================
# HYBRID RETRIEVAL
# =========================
def retrieve(query):
    q_vec = model.encode(query, normalize_embeddings=True)

    D, I = index.search(q_vec.reshape(1, -1), TOP_K)

    faiss_results = [paper_ids[i] for i in I[0]]
    citation_weights = normalize_scores(faiss_results)

    scored = []
    for i, pid in enumerate(faiss_results):
        faiss_score = float(D[0][i])
        citation_w = citation_weights.get(pid, 0)

        score = ALPHA * faiss_score + BETA * citation_w
        scored.append((score, pid))

    scored.sort(reverse=True)
    return [p for _, p in scored]

# =========================
# GRAPH EXPANSION
# =========================
def expand(seed):
    visited = set(seed)
    frontier = set(seed)

    for _ in range(HOPS):
        new = set()
        for n in frontier:
            new.update(graph[n])
        visited.update(new)
        frontier = new

    return visited

# =========================
# CONTEXT BUILDER
# =========================
def build_context(seeds):
    nodes = expand(seeds)
    texts = []

    for pid in nodes:
        if pid in papers:
            texts.append(f"Paper {pid}: {decode(papers[pid])}")

    return "\n".join(texts)

# =========================
# ASK
# =========================
def ask(q):
    seeds = retrieve(q)
    context = build_context(seeds)

    prompt = f"""
You are a scientific research assistant.

Use ONLY the context.

QUESTION:
{q}

CONTEXT:
{context}

Answer clearly and explain relationships between papers.
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
    print("Hybrid GraphRAG ready 🚀 (exit to quit)\n")

    while True:
        q = input("Question: ")
        if q == "exit":
            break

        print("\nAnswer:\n")
        print(ask(q))
        print("\n" + "-"*50 + "\n")