# GraphRAG for CORA Dataset 
### (FAISS + Citation Graph + LLM)

A **hybrid GraphRAG system** for scientific paper retrieval and reasoning over the CORA dataset.  
This project combines:

- Dense retrieval (FAISS + SentenceTransformers)
- Citation graph expansion
- Graph ranking (PageRank-style)
- LLM reasoning (Ollama / Gemma)
- Hybrid reranking (semantic + graph importance)

---

## Overview

Traditional RAG systems rely only on vector similarity.  
This project enhances retrieval using **graph structure (citations)** to improve relevance and reasoning.

### Key Idea

> Combine **semantic similarity + graph connectivity + LLM reasoning** for better scientific paper understanding.

---

## System Architecture

```
User Query - > Sentence Embedding (MiniLM) - > FAISS Retrieval (Top-K Papers) - > Citation Graph Expansion (Multi-hop) - > PageRank / Hybrid Scoring - > Context Construction - > LLM (Ollama - Gemma3) - > Final Answer
```

---

## Project Structure

```
GraphRAG/
│
├── CORA/ # Raw dataset files
│ ├── papers_dataset.txt
│ ├── citations.txt
│ ├── words_dictionary.txt
│ ├── authors_dataset.txt
│
├── paper_embeddings.npy # Precomputed embeddings
├── paper_ids.npy # Paper ID mapping
├── faiss_index.bin # FAISS index (optional)
│
├── build_embeddings.py # Build text embeddings
├── build_index.py # Build FAISS index
├── graphrag_hybrid.py # Hybrid GraphRAG pipeline
├── graphrag_ultra.py # Advanced version (PageRank + rerank)
│
└── README.md
```

---

## Installation

### 1. Clone repository

```bash
git clone https://github.com/your-username/graphrag-cora.git
```
then
```
cd graphrag-cora
```

2. Install dependencies

```
pip install faiss-cpu sentence-transformers numpy ollama

```

3. Install Ollama (for LLM)


https://ollama.com


Then pull a model:

```
ollama pull gemma3:4b
```


## Pipeline Steps:

1. Build Embeddings

```
python build_embeddings.py
```

Generates:

```
paper_embeddings.npy
paper_ids.npy
```

2. Build FAISS Index

```
python build_index.py
```

Creates a similarity search index for papers.

3. Run GraphRAG System

```
python graphrag_hybrid.py
```

or advanced version:

```
python graphrag_ultra.py
```

## Example Queries: 

Query 1: Neural Networks

```
Which papers are related to neural networks?
```

Output:

```
Papers about backpropagation
VC dimension theory
Perceptrons
Adaptive neural systems
```

Query 2: Data Mining

```
Which papers are related to data mining?
```

Output:
```
Association rule mining
Feature selection
Clustering (BIRCH)
Decision trees
```

Query 3: Social Networks

```
Which papers are related to social networks?
```
Output:

```
Graph theory papers
Network routing
Digraph analysis
Distributed systems
```

## Key Features: 
- Hybrid Retrieval
- FAISS semantic search
- Citation graph importance

## Graph Expansion

- Multi-hop neighbor exploration
- Citation-based context enrichment

## Ranking Strategy

```
Final Score = α * FAISS + β * PageRank
```

## LLM Reasoning

- Uses Ollama (Gemma / LLaMA)
- Context-aware scientific answers

## Technologies Used
- Python 
- FAISS (Facebook AI Similarity Search)
- SentenceTransformers (MiniLM)
- NetworkX-style citation graph
- Ollama (LLM inference)
- NumPy

## Future Improvements
 - Cross-Encoder reranking
 - Query expansion (multi-query retrieval)
 - Context compression (LLM summarization)
 - Personalized PageRank per query

 ## Evaluation metrics 

- Recall@K
- MRR

## Research Motivation

This project demonstrates how graph structure + neural retrieval + LLM reasoning can significantly improve:

- Scientific paper search
- Knowledge graph QA
- Academic recommendation systems