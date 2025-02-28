import faiss
import numpy as np

index = faiss.read_index('app/rag_data/index.faiss')
embeddings = np.load('app/rag_data/embeddings.npy')
query = embeddings[0:1]  # Usar o primeiro embedding como consulta
distances, indices = index.search(query, 5)
print(f"Distâncias: {distances}, Índices: {indices}")