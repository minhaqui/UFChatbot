import os
import json
import numpy as np
import faiss
import google.generativeai as genai
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import logging
import hashlib
import time
from google.cloud import storage
import sys

# Configurações
EMBEDDED_DIR = "/app/rag_data"
CHUNKS_JSON = os.path.join(EMBEDDED_DIR, "chunks.json")
INDEX_PATH = os.path.join(EMBEDDED_DIR, "index.faiss")
EMBEDDINGS_PATH = os.path.join(EMBEDDED_DIR, "embeddings.npy")
CACHE_FILE = os.path.join(EMBEDDED_DIR, "embedding_cache.json")
LOG_FILE = "embedding_debug.log"
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "seu-bucket-aqui")

# Configurar logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configurar API Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY não definida nas variáveis de ambiente.")
genai.configure(api_key=GEMINI_API_KEY)

def load_chunks(json_path):
    """Carrega os chunks de um arquivo JSON local ou do GCS."""
    if 'CLOUD_RUN' in os.environ:
        client = storage.Client()
        bucket = client.get_bucket(GCS_BUCKET_NAME)
        blob = bucket.blob("chunks.json")
        json_content = blob.download_as_string().decode('utf-8')
        chunks = json.loads(json_content)
    else:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Arquivo {json_path} não encontrado.")
        with open(json_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    
    logging.info(f"Carregados {len(chunks)} chunks.")
    print(f"Carregados {len(chunks)} chunks.")
    return chunks

def load_cache():
    """Carrega o cache de embeddings local ou do GCS."""
    if 'CLOUD_RUN' in os.environ:
        client = storage.Client()
        bucket = client.get_bucket(GCS_BUCKET_NAME)
        blob = bucket.blob("embedding_cache.json")
        if blob.exists():
            cache_content = blob.download_as_string().decode('utf-8')
            return json.loads(cache_content)
        return {}
    else:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

def save_cache(cache):
    """Salva o cache de embeddings local ou no GCS."""
    cache_json = json.dumps(cache, ensure_ascii=False, indent=4)
    if 'CLOUD_RUN' in os.environ:
        client = storage.Client()
        bucket = client.get_bucket(GCS_BUCKET_NAME)
        blob = bucket.blob("embedding_cache.json")
        blob.upload_from_string(cache_json, content_type='application/json')
    else:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            f.write(cache_json)
    logging.info(f"Cache salvo.")

def save_to_gcs(local_path, gcs_path):
    """Salva um arquivo local no Google Cloud Storage."""
    if 'CLOUD_RUN' in os.environ:
        client = storage.Client()
        bucket = client.get_bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        logging.info(f"Arquivo salvo no GCS: {gcs_path}")
        print(f"Arquivo salvo no GCS: {gcs_path}")

def get_chunk_hash(chunk):
    """Gera um hash MD5 do chunk para verificar alterações."""
    return hashlib.md5(chunk.encode('utf-8')).hexdigest()

def generate_embedding_single(chunk, cache):
    """Gera embedding para um único chunk com cache."""
    chunk_hash = get_chunk_hash(chunk)
    if chunk_hash in cache:
        logging.debug(f"Embedding de chunk reutilizado do cache: {chunk[:50]}...")
        return np.array(cache[chunk_hash], dtype='float32')

    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=chunk,
            task_type="retrieval_document"
        )
        embedding = response['embedding']
        cache[chunk_hash] = embedding
        return np.array(embedding, dtype='float32')
    except Exception as e:
        logging.error(f"Erro ao gerar embedding para chunk: {e}")
        return np.zeros(768, dtype='float32')

def generate_embeddings_gemini_api(chunks, cache):
    """Gera embeddings usando a API Gemini em paralelo com progresso."""
    logging.info(f"Iniciando geração de embeddings para {len(chunks)} chunks.")
    print(f"Iniciando geração de embeddings para {len(chunks)} chunks.")
    start_time = time.time()

    num_processes = 20 if 'CLOUD_RUN' not in os.environ else min(cpu_count(), 4)
    with Pool(num_processes) as pool:
        embeddings_list = list(tqdm(
            pool.starmap(generate_embedding_single, [(chunk, cache) for chunk in chunks]),
            total=len(chunks),
            desc="Gerando embeddings"
        ))

    duration = time.time() - start_time
    logging.info(f"Embeddings gerados em {duration:.2f} segundos com {num_processes} processos.")
    print(f"Embeddings gerados em {duration:.2f} segundos com {num_processes} processos.")
    return np.array(embeddings_list, dtype='float32')

def build_index(embeddings):
    """Constrói e retorna o índice FAISS IndexFlatL2, usando GPU se disponível."""
    dimension = embeddings.shape[1]
    if 'USE_FAISS_GPU' in os.environ and os.environ['USE_FAISS_GPU'].lower() == 'true':
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, dimension)
    else:
        index = faiss.IndexFlatL2(dimension)
    
    index.add(embeddings)
    logging.info(f"Índice FAISS construído com {index.ntotal} vetores (GPU: {'USE_FAISS_GPU' in os.environ}).")
    print(f"Índice FAISS construído com {index.ntotal} vetores (GPU: {'USE_FAISS_GPU' in os.environ}).")
    return index

if __name__ == "__main__":
    if 'CLOUD_RUN' not in os.environ:
        os.makedirs(EMBEDDED_DIR, exist_ok=True)

    try:
        chunks = load_chunks(CHUNKS_JSON)
    except Exception as e:
        logging.error(f"Erro ao carregar chunks: {e}")
        print(f"Erro ao carregar chunks: {e}")
        sys.exit(1)

    if not chunks:
        logging.warning("Nenhum chunk carregado. Finalizando execução.")
        print("Nenhum chunk carregado. Finalizando execução.")
        sys.exit(1)

    cache = load_cache()

    embeddings = generate_embeddings_gemini_api(chunks, cache)
    if len(embeddings) != len(chunks):
        logging.warning(f"Inconsistência: {len(embeddings)} embeddings gerados para {len(chunks)} chunks.")
        print(f"Inconsistência: {len(embeddings)} embeddings gerados para {len(chunks)} chunks.")
    else:
        logging.info(f"Gerados {len(embeddings)} embeddings com sucesso.")
        print(f"Gerados {len(embeddings)} embeddings com sucesso.")

    index = build_index(embeddings)
    
    # Converter índice GPU para CPU antes de salvar
    if 'USE_FAISS_GPU' in os.environ and os.environ['USE_FAISS_GPU'].lower() == 'true':
        index = faiss.index_gpu_to_cpu(index)
        logging.info("Índice convertido de GPU para CPU para salvamento.")
        print("Índice convertido de GPU para CPU para salvamento.")
    
    faiss.write_index(index, INDEX_PATH)
    np.save(EMBEDDINGS_PATH, embeddings)
    save_cache(cache)

    if 'CLOUD_RUN' in os.environ:
        save_to_gcs(INDEX_PATH, "index.faiss")
        save_to_gcs(EMBEDDINGS_PATH, "embeddings.npy")

    logging.info(f"Índice FAISS salvo em {INDEX_PATH}, embeddings em {EMBEDDINGS_PATH}.")
    print(f"Índice FAISS salvo em {INDEX_PATH}, embeddings em {EMBEDDINGS_PATH}.")