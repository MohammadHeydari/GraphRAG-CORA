import ollama
from collections import defaultdict

# =========================
# CONFIG
# =========================
BASE_PATH = "./CORA/"
TOP_K_SEEDS = 3
HOPS = 1

# =========================
# 1. LOAD WORDS DICTIONARY
# =========================
words_dict = {}

with open(BASE_PATH + "words_dictionary.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        word, code = line.split()
        words_dict[code] = word


# =========================
# 2. LOAD PAPERS
# =========================
papers = {}

with open(BASE_PATH + "papers_dataset.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(";")
        if len(parts) < 3:
            continue
        paper_id = parts[0]
        features = parts[2]
        papers[paper_id] = features


# =========================
# 3. LOAD CITATION GRAPH
# =========================
citation_graph = defaultdict(list)

with open(BASE_PATH + "citations.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        inside = line.split("(")[1].split(")")[0]
        src, dst = inside.split(",")
        citation_graph[src].append(dst)


# =========================
# 4. DECODE FEATURES
# =========================
def decode_features(feature_string):
    words = []
    for item in feature_string.split(","):
        code = item.split(":")[0]
        if code in words_dict:
            words.append(words_dict[code])
    return " ".join(words)


# =========================
# 5. SIMPLE RETRIEVAL (IMPORTANT FIX)
# =========================
def score_paper(query, paper_text):
    query_words = set(query.lower().split())
    paper_words = set(paper_text.lower().split())
    return len(query_words & paper_words)


def retrieve_seed_papers(query, top_k=TOP_K_SEEDS):
    scores = []

    for pid, feat in papers.items():
        text = decode_features(feat)
        score = score_paper(query, text)
        if score > 0:
            scores.append((score, pid))

    scores.sort(reverse=True)
    return [pid for _, pid in scores[:top_k]]


# =========================
# 6. GRAPH EXPANSION
# =========================
def get_neighbors(paper_ids, hops=HOPS):
    visited = set(paper_ids)
    frontier = set(paper_ids)

    for _ in range(hops):
        new_frontier = set()

        for node in frontier:
            for neigh in citation_graph.get(node, []):
                if neigh not in visited:
                    new_frontier.add(neigh)

        visited.update(new_frontier)
        frontier = new_frontier

    return visited


# =========================
# 7. BUILD CONTEXT
# =========================
def build_context(seed_papers):
    neighbors = get_neighbors(seed_papers)

    texts = []
    for pid in neighbors:
        if pid in papers:
            decoded = decode_features(papers[pid])
            texts.append(f"Paper {pid}: {decoded}")

    return "\n".join(texts)


# =========================
# 8. GRAPH RAG + OLLAMA
# =========================
def ask(question):
    seeds = retrieve_seed_papers(question)

    if not seeds:
        return "No relevant papers found in CORA dataset."

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


# =========================
# 9. CLI LOOP
# =========================
if __name__ == "__main__":
    print("GraphRAG CORA ready (AUTO-SEED MODE)! type 'exit' to quit\n")

    while True:
        q = input("Question: ")
        if q.lower() == "exit":
            break

        print("\nAnswer:\n")
        print(ask(q))
        print("\n" + "-" * 50 + "\n")