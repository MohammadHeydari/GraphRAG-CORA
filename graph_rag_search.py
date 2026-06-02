# graph_rag_search.py

import numpy as np
import faiss
from torch_geometric.datasets import Planetoid


# Load dataset
dataset = Planetoid(root='data', name='Cora')
data = dataset[0]
edges = data.edge_index.numpy()


# Load embeddings
embeddings = np.load("embeddings.npy")


# Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)



# GraphRAG multi-hop retrieval
def graph_rag_search(query_vec, k=5):
    # Step 1: vector retrieval
    D, I = index.search(query_vec, k)
    top_nodes = I[0]

    # Step 2: expand via neighbors
    expanded = set(top_nodes)
    for n in top_nodes:
        nbors = edges[1, edges[0] == n]  # outgoing edges
        expanded.update(nbors)

    return list(expanded)



# Test
query = embeddings[0].reshape(1, -1)
result_nodes = graph_rag_search(query)
print("Expanded nodes for GraphRAG:", result_nodes)