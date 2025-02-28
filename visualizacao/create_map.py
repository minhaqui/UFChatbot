# create_map.py

import os
import json
import subprocess
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import google.generativeai as genai

# Configuração da chave de API do Gemini
genai.configure(api_key="OPENROUTE_API_KEY")

# Definição dos modelos
KEYWORD_MODEL = "deepseek-r1:latest"  # Modelo Ollama para palavras-chave
CLUSTER_SUMMARY_MODEL = "google/gemini-2.0-flash-thinking-exp:free"  # Modelo Gemini para resumos de clusters

# Arquivo para salvar resultados parciais (keywords)
KEYWORDS_FILE = "keywords_partial.json"
# Limite de caracteres para o texto combinado dos clusters
MAX_CLUSTER_TEXT_LENGTH = 10000  # Ajuste conforme necessário

def run_ollama(model_name: str, prompt: str, timeout: int = 30) -> str:
    """
    Executa o comando "ollama run <model_name> <prompt>" de forma síncrona e retorna a saída.
    """
    try:
        proc = subprocess.run(
            ["ollama", "run", model_name, prompt],
            capture_output=True,
            encoding="utf-8",
            timeout=timeout
        )
        output = proc.stdout.strip()
        if not output:
            err = proc.stderr.strip()
            print(f"[AVISO] Modelo '{model_name}' retornou saída vazia. stderr: {err}")
            return ""
        return output
    except subprocess.TimeoutExpired:
        print(f"[ERRO] Timeout ao chamar modelo '{model_name}'")
        return ""
    except UnicodeDecodeError as ude:
        print(f"[ERRO] Falha na decodificação da saída do modelo '{model_name}': {ude}")
        return ""
    except Exception as e:
        print(f"[EXCEÇÃO] Erro ao chamar modelo '{model_name}': {e}")
        return ""

def get_representative_keyword(chunk_text: str) -> str:
    """
    Gera uma palavra-chave (até 2 palavras) para o chunk usando o modelo Ollama.
    """
    prompt = f"Responda com até 2 palavra-chave que resumam o texto.\nTexto: {chunk_text}"
    result = run_ollama(KEYWORD_MODEL, prompt, timeout=30)
    return result if result else "N/A"

def get_cluster_summary(cluster_texts: list) -> str:
    """
    Gera um resumo (até 2 frases) do tema central do cluster usando o modelo Gemini.
    """
    combined_text = "\n".join(cluster_texts)
    if len(combined_text) > MAX_CLUSTER_TEXT_LENGTH:
        print(f"[AVISO] Texto do cluster muito longo ({len(combined_text)} caracteres), truncando para {MAX_CLUSTER_TEXT_LENGTH}...")
        combined_text = combined_text[:MAX_CLUSTER_TEXT_LENGTH]
    
    prompt = f"Responda com até 2 frases que descrevam o tema central dos textos.\nTextos: {combined_text}"
    
    try:
        model = genai.GenerativeModel(CLUSTER_SUMMARY_MODEL)
        response = model.generate_content(prompt)
        summary = response.text.strip()
        return summary if summary else "N/A"
    except Exception as e:
        print(f"[ERRO] Falha ao chamar o Gemini: {e}")
        return "N/A"

def load_partial_keywords() -> dict:
    """
    Carrega keywords parciais do arquivo, se existir.
    """
    if os.path.exists(KEYWORDS_FILE):
        try:
            with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERRO] Não foi possível carregar {KEYWORDS_FILE}: {e}")
    return {}

def save_partial_keywords(keywords_dict: dict):
    """
    Salva keywords parciais no arquivo.
    """
    try:
        with open(KEYWORDS_FILE, "w", encoding="utf-8") as f:
            json.dump(keywords_dict, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"[ERRO] Não foi possível salvar {KEYWORDS_FILE}: {e}")

def process_chunk(index: int, chunk: str, keywords_dict: dict):
    """
    Processa um chunk e gera sua palavra-chave.
    """
    keyword = get_representative_keyword(chunk)
    keywords_dict[str(index)] = keyword
    print(f"[INFO] Chunk {index+1}: keyword='{keyword}'")
    save_partial_keywords(keywords_dict)

def main():
    embeddings_file = "embeddings.npy"
    chunks_file = "chunks.json"

    # Verifica se os arquivos existem
    if not os.path.exists(embeddings_file):
        print(f"[ERRO] Arquivo {embeddings_file} não encontrado em {os.getcwd()}")
        return
    if not os.path.exists(chunks_file):
        print(f"[ERRO] Arquivo {chunks_file} não encontrado em {os.getcwd()}")
        return

    print("[INFO] Carregando embeddings...")
    embeddings = np.load(embeddings_file)
    print("[INFO] Carregando chunks...")
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Sincroniza embeddings e chunks
    if len(embeddings) != len(chunks):
        print(f"[AVISO] Número de embeddings ({len(embeddings)}) difere de chunks ({len(chunks)}). Sincronizando...")
        n = min(len(embeddings), len(chunks))
        embeddings = embeddings[:n]
        chunks = chunks[:n]

    # Carrega resultados parciais
    keywords_dict = load_partial_keywords()
    total = len(chunks)
    print(f"[INFO] Precisamos gerar keywords para {total} chunks (já temos {len(keywords_dict)}).")

    # Processa os chunks sequencialmente
    for i, chunk in enumerate(chunks):
        if str(i) not in keywords_dict:
            process_chunk(i, chunk, keywords_dict)

    keywords = [keywords_dict.get(str(i), "N/A") for i in range(len(chunks))]

    print("[INFO] Aplicando PCA para redução dimensional em 3D...")
    pca = PCA(n_components=3, random_state=42)
    embeddings_3d = pca.fit_transform(embeddings)

    n_clusters = 5
    print(f"[INFO] Aplicando K-Means com {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_3d)

    # Gera resumo para cada cluster usando Gemini
    cluster_summaries = {}
    for cluster_id in range(n_clusters):
        cluster_texts = [chunks[i] for i in range(len(chunks)) if cluster_labels[i] == cluster_id]
        print(f"[INFO] Resumindo cluster {cluster_id} (n={len(cluster_texts)})...")
        summary = get_cluster_summary(cluster_texts)
        cluster_summaries[cluster_id] = summary

    # Salva os resultados
    df = pd.DataFrame(embeddings_3d, columns=["x", "y", "z"])
    df["keyword"] = keywords
    df["chunk_text"] = chunks
    df["cluster"] = cluster_labels
    df.to_csv("map_data.csv", index=False)
    print("[INFO] Mapa salvo em 'map_data.csv'.")

    with open("cluster_summaries.json", "w", encoding="utf-8") as f:
        json.dump(cluster_summaries, f, ensure_ascii=False, indent=4)
    print("[INFO] Resumos dos clusters salvos em 'cluster_summaries.json'.")

if __name__ == "__main__":
    main()