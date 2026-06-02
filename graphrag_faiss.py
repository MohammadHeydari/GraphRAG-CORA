# graphrag_faiss.py
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import ollama
import os

BASE_PATH = "./CORA/"
TOP_K = 5
HOPS = 1


# LOAD PAPERS
papers = {}
with open(os.path.join(BASE_PATH, "papers_dataset.txt"), "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(";")
        if len(parts) < 3:
            continue
        paper_id = parts[0]
        features = parts[2]
        papers[paper_id] = features


# WORDS DICTIONARY
words_dict = {}
with open(os.path.join(BASE_PATH, "words_dictionary.txt"), "r", encoding="utf-8") as f:
    for line in f:
        word, code = line.strip().split()
        words_dict[code] = word

def decode_features(feature_string):
    words = []
    for item in feature_string.split(","):
        code = item.split(":")[0]
        if code in words_dict:
            words.append(words_dict[code])
    return " ".join(words)


# LOAD CITATION GRAPH

citation_graph = defaultdict(list)
with open(os.path.join(BASE_PATH, "citations.txt"), "r", encoding="utf-8") as f:
    for line in f:
        inside = line.strip().split("(")[1].split(")")[0]
        src, dst = inside.split(",")
        citation_graph[src].append(dst)


# LOAD FAISS + EMBEDDINGS

embeddings = np.load("paper_embeddings.npy")
paper_ids = np.load("paper_ids.npy")
dim = embeddings.shape[1]
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

model = SentenceTransformer("all-MiniLM-L6-v2")

# RETRIEVAL

def retrieve_top_papers(query, k=TOP_K):
    query_vec = model.encode(query, normalize_embeddings=True)
    D, I = index.search(query_vec.reshape(1, -1), k)
    return [paper_ids[i] for i in I[0]]


# GRAPH EXPANSION

def get_neighbors(seed_papers, hops=HOPS):
    visited = set(seed_papers)
    frontier = set(seed_papers)
    for _ in range(hops):
        new_frontier = set()
        for node in frontier:
            for neigh in citation_graph.get(node, []):
                if neigh not in visited:
                    new_frontier.add(neigh)
        visited.update(new_frontier)
        frontier = new_frontier
    return visited

# BUILD CONTEXT

def build_context(seed_papers):
    neighbors = get_neighbors(seed_papers)
    texts = []
    for pid in neighbors:
        if pid in papers:
            decoded = decode_features(papers[pid])
            texts.append(f"Paper {pid}: {decoded}")
    return "\n".join(texts)

# ASK LLM

def ask(question):
    seeds = retrieve_top_papers(question)
    if not seeds:
        return "No relevant papers found."
    context = build_context(seeds)
    prompt = f"""
You are a scientific research assistant.

Answer the question using ONLY the provided context.

QUESTION:
{question}

CONTEXT:
{context}

Instructions:
- Be precise
- Use citations if possible
- Focus on relationships between papers

Answer:
"""
    response = ollama.chat(
        model="gemma3:4b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# CLI

if __name__ == "__main__":
    print("GraphRAG CORA PRO ready! type 'exit' to quit\n")
    while True:
        q = input("Question: ")
        if q.lower() == "exit":
            break
        print("\nAnswer:\n")
        print(ask(q))
        print("\n" + "-"*50 + "\n")