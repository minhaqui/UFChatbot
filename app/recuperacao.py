import os
import numpy as np
import faiss
import google.generativeai as genai
import json
import requests
import logging
from google.api_core import exceptions

# Configurar variáveis de ambiente para silenciar logs do gRPC
os.environ["GRPC_VERBOSITY"] = "ERROR"  # Define o nível de log do gRPC para ERROR
os.environ["GRPC_TRACE"] = ""  # Desativa tracing do gRPC

# Configurar logging do Python
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configurações
EMBEDDED_DIR = "rag_data"
INDEX_PATH = os.path.join(EMBEDDED_DIR, "index.faiss")
EMBEDDINGS_PATH = os.path.join(EMBEDDED_DIR, "embeddings.npy")
CHUNKS_JSON = os.path.join(EMBEDDED_DIR, "chunks.json")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "google/gemini-2.0-flash-lite-preview-02-05:free"
GEMINI_EMBEDDING_MODEL_NAME = "models/embedding-001"

def configure_gemini():
    if not GEMINI_API_KEY:
        raise ValueError("A variável GEMINI_API_KEY não está definida.")
    genai.configure(api_key=GEMINI_API_KEY)
    logging.info("Usando Gemini API para query embeddings.")

def load_chunks():
    if not os.path.exists(CHUNKS_JSON):
        raise FileNotFoundError(f"Arquivo {CHUNKS_JSON} não encontrado.")
    with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def load_faiss_index():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Arquivo {INDEX_PATH} não encontrado.")
    return faiss.read_index(INDEX_PATH)

def load_embeddings():
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(f"Arquivo {EMBEDDINGS_PATH} não encontrado.")
    return np.load(EMBEDDINGS_PATH)

def get_query_embedding(query, embeddings=None):
    if embeddings is not None:
        logging.info("Usando embedding mockado para teste (primeiro embedding).")
        return embeddings[0:1]
    configure_gemini()
    try:
        response = genai.embed_content(
            model=GEMINI_EMBEDDING_MODEL_NAME,
            content=query,
            task_type="retrieval_document",
            title="Query Embedding"
        )
        return np.array([response['embedding']], dtype='float32')
    except exceptions.GoogleAPIError as e:
        logging.error(f"Erro na API Gemini: {e}")
        raise
    except Exception as e:
        logging.error(f"Erro inesperado ao gerar embedding: {e}")
        raise

def search(query, index, texts, embeddings=None, top_k=10):
    try:
        query_embedding = get_query_embedding(query, embeddings)
        distances, indices = index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(texts):
                results.append((texts[idx], dist))
            else:
                logging.warning(f"Índice {idx} fora do intervalo de textos.")
        return results
    except Exception as e:
        logging.error(f"Erro na busca FAISS: {e}")
        raise

def search_chunks(query, top_k=10, threshold=0.5):
    index = faiss.read_index("app/rag_data/index.faiss")
    with open("app/rag_data/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    from models import embed_query
    query_embedding = embed_query(query)
    distances, indices = index.search(query_embedding, top_k)
    relevant_chunks = [chunks[idx] for idx, dist in zip(indices[0], distances[0]) if dist < threshold]
    return relevant_chunks if relevant_chunks else ["Nenhum contexto relevante encontrado."]

def build_prompt(query, results):
    prompt = f"Consulta: {query}\n\nContexto recuperado:\n"
    for i, (text, distance) in enumerate(results):
        prompt += f"\n--- Resultado {i+1} (distância: {distance:.4f}) ---\n{text}\n"
    prompt += "\nCom base nesse contexto, responda a consulta de forma clara e concisa."
    return prompt

def get_openrouter_response(prompt):
    if not OPENROUTER_API_KEY:
        raise ValueError("A variável OPENROUTER_API_KEY não está definida.")
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": OPENROUTER_MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 200}
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        logging.error(f"Erro ao chamar OpenRouter: {e}")
        raise

if __name__ == "__main__":
    try:
        texts = load_chunks()
        index = load_faiss_index()
        embeddings = load_embeddings()  # Carrega embeddings para fallback
        query = "Quais são os requisitos para participar de uma licitação pública?"
        
        # Tenta usar Gemini, com fallback para mock no Windows
        try:
            results = search(query, index, texts, top_k=10)
        except Exception as e:
            logging.warning("Falha ao usar Gemini API, usando embedding mockado.")
            results = search(query, index, texts, embeddings, top_k=10)
        
        prompt = build_prompt(query, results)
        print(prompt)
        
        response = get_openrouter_response(prompt)
        print("\nResposta Automática (OpenRouter):")
        print(response)
    except Exception as e:
        logging.error(f"Erro durante a execução: {e}")
        print(f"Erro: Veja o log para detalhes.")