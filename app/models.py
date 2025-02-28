import requests
import faiss
import json
import numpy as np
import os
import google.generativeai as genai
from google.api_core import exceptions
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Configuração do modelo
MODEL = "google/gemini-2.0-flash-001"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

print(genai.__version__)
# Carregar índice FAISS e chunks
index_path = os.path.join(os.path.dirname(__file__), "rag_data", "index.faiss")
index = faiss.read_index(index_path)
with open("app/rag_data/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

cache_x = {}
cache_y = {}

def embed_query(query):
    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_document"
        )
        return np.array([response['embedding']], dtype='float32')
    except exceptions.GoogleAPIError as e:
        logger.error(f"Erro ao gerar embedding: {e}")
        return np.zeros((1, 768), dtype='float32')

def generate_response(query, context_chunks=None):
    if context_chunks:
        context = " ".join(context_chunks)
        prompt = f"""     
            "Seu nome é Eniac Jr.\n"
            "Você é um especialista em contratações públicas no Brasil bastante experiente, "
            "especialmente treinado na lei 14.133/2021 e suas aplicações.\n"
            "Responda a pergunta em português formal, claro e conciso, usando bulletpoints quando couber.\n"
            "Elabore uma resposta que possua uma breve introdução e ao final indique como o usuário pode se aprofundar sobre sua dúvida ou problema.\n"
            "Não responda sobre o tipo de modelo de LLM você é ou quais tecnologias está usando.\n"
            "Não responda sobre outros assuntos que não envolvam, direta ou indiretamente, contratações públicas no Brasil."
            Forneça uma resposta curta à consulta: {query}\nContexto fornecido: {context}"
            """
    else:
        prompt = f"""
            "Seu nome é Eniac Jr.\n"
            "Você é um especialista em contratações públicas no Brasil bastante experiente, "
            "especialmente treinado na lei 14.133/2021 e suas aplicações.\n"
            "Responda a pergunta em português formal, claro e conciso, usando bulletpoints quando couber.\n"
            "Elabore uma resposta que possua uma breve introdução e ao final indique como o usuário pode se aprofundar sobre sua dúvida ou problema.\n"
            "Não responda sobre o tipo de modelo de LLM você é ou quais tecnologias está usando.\n"
            "Não responda sobre outros assuntos que não envolvam, direta ou indiretamente, contratações públicas no Brasil."
            Forneça uma resposta curta à consulta: {query}"
            """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        response_json = response.json()
        if 'error' in response_json:
            error_msg = response_json['error']
            if isinstance(error_msg, dict):
                error_msg = error_msg.get('message', 'Erro desconhecido')
            return f"Erro: {error_msg}"
        if 'choices' in response_json:
            return response_json["choices"][0]["message"]["content"]
        elif 'content' in response_json:
            return response_json['content']
        elif 'text' in response_json:
            return response_json['text']
        else:
            return "Erro: Resposta da API em formato inesperado."
    except Exception as e:
        if "timed out" in str(e).lower():
            return "Erro: Request timeout"
        try:
            fallback_payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}]
            }
            fallback_response = requests.post(OPENROUTER_API_URL, headers=headers, json=fallback_payload, timeout=10)
            fallback_response.raise_for_status()
            fallback_json = fallback_response.json()
            if 'choices' in fallback_json:
                return fallback_json["choices"][0]["message"]["content"]
            else:
                return "Resposta padrão: modelo indisponível no momento."
        except Exception:
            return "Resposta padrão: modelo indisponível no momento."
        
def modelo_x_response(query, historico):
    # Cria uma chave única com base no query e no histórico
    key = (query, tuple((msg['remetente'], msg['conteudo']) for msg in historico))
    if key in cache_x:
        return cache_x[key]
    
    # Constrói o prompt com histórico
    prompt = ""
    for msg in historico:
        if msg['remetente'] == 'user':
            prompt += f"Usuário: {msg['conteudo']}\n"
        else:
            prompt += f"Modelo X: {msg['conteudo']}\n"
    prompt += f"Usuário: {query}\nModelo X: "
    
    resposta = generate_response(prompt)
    cache_x[key] = resposta
    return resposta

def modelo_y_response(query, historico):
    from app.recuperacao import search_chunks  
    # Importa a função de RAG
    key = (query, tuple((msg['remetente'], msg['conteudo']) for msg in historico))
    if key in cache_y:
        return cache_y[key]
    
    context_chunks = search_chunks(query)
    prompt = "Contexto: " + " ".join(context_chunks) + "\n"
    for msg in historico:
        if msg['remetente'] == 'user':
            prompt += f"Usuário: {msg['conteudo']}\n"
        else:
            prompt += f"Modelo Y: {msg['conteudo']}\n"
    prompt += f"Usuário: {query}\nModelo Y: "
    
    resposta = generate_response(prompt)
    cache_y[key] = resposta
    return resposta