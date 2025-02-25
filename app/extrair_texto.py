import os
import re
import json
import PyPDF2
import spacy
import logging
from pdf2image import convert_from_path
import pytesseract
from PyPDF2.errors import PdfReadError
from multiprocessing import Pool, cpu_count
from docx import Document
import pandas as pd
import docx2txt
import hashlib

# Configurações
MAX_CHUNK_TOKENS = 800
MIN_CHUNK_TOKENS = 100
OVERLAP_TOKENS = 150
MAX_CHUNK_BYTES = 9000
LOG_FILE = 'chunking_debug.log'
TESSERACT_LANG = 'por'
CACHE_FILE = "app/rag_data/chunk_cache.json"  # Novo: cache de arquivos processados

# Carregar SpaCy com sentencizer
NLP = spacy.load("pt_core_news_sm", disable=["ner", "parser"])
NLP.add_pipe('sentencizer')

# Configuração do logging (menos verboso)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_cache():
    """Carrega o cache de arquivos já processados."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    """Salva o cache de arquivos processados."""
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)

def get_file_hash(file_path):
    """Gera um hash MD5 do arquivo para verificar se mudou."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num in range(len(reader.pages)):
                try:
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text += page_text + "\n"
                    else:
                        text += extract_text_from_image(file_path, page_num)
                except PdfReadError:
                    text += extract_text_from_image(file_path, page_num)
    except Exception as e:
        logging.error(f"Erro ao processar PDF {file_path}: {e}")
    return text

def extract_text_from_image(file_path, page_num):
    try:
        images = convert_from_path(file_path, first_page=page_num + 1, last_page=page_num + 1)
        return "".join(pytesseract.image_to_string(image, lang=TESSERACT_LANG) + "\n" for image in images)
    except Exception as e:
        logging.error(f"Erro ao aplicar OCR na página {page_num + 1} de {file_path}: {e}")
        return ""

def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        logging.error(f"Erro ao processar .docx {file_path}: {e}")
        return ""

def extract_text_from_doc(file_path):
    try:
        return docx2txt.process(file_path)
    except Exception as e:
        logging.error(f"Erro ao processar .doc {file_path}: {e}")
        return ""

def extract_text_from_excel(file_path):
    try:
        xls = pd.ExcelFile(file_path)
        text = ""
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            text += "\n".join(df.astype(str).apply(lambda x: " ".join(x), axis=1)) + "\n"
        return text
    except Exception as e:
        logging.error(f"Erro ao processar Excel {file_path}: {e}")
        return ""

def extract_text_from_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logging.error(f"Erro ao processar .txt {file_path}: {e}")
        return ""

def get_file_type(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    return {
        ".pdf": "pdf",
        ".docx": "docx",
        ".doc": "doc",
        ".xls": "excel",
        ".xlsx": "excel",
        ".txt": "txt"
    }.get(ext, "unknown")

def extract_text_from_file(file_path):
    file_type = get_file_type(file_path)
    if file_type == "pdf":
        return extract_text_from_pdf(file_path)
    elif file_type == "docx":
        return extract_text_from_docx(file_path)
    elif file_type == "doc":
        return extract_text_from_doc(file_path)
    elif file_type == "excel":
        return extract_text_from_excel(file_path)
    elif file_type == "txt":
        return extract_text_from_txt(file_path)
    else:
        logging.warning(f"Tipo de arquivo desconhecido: {file_path}")
        return ""

def preprocess_text(text):
    text = re.sub(r'Página \d+ de \d+', '', text)
    text = re.sub(r'\n\d+\n', '\n', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def semantic_segmentation_with_overlap(text):
    try:
        doc = NLP(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except Exception as e:
        logging.error(f"Erro ao processar texto com SpaCy: {e}")
        return []

    chunks = []
    current_chunk = []
    current_token_count = 0

    for idx, sentence in enumerate(sentences):
        sentence_tokens = len(list(NLP(sentence)))
        if current_token_count + sentence_tokens > MAX_CHUNK_TOKENS:
            chunk_text = " ".join(current_chunk)
            chunk_bytes = len(chunk_text.encode('utf-8'))
            if chunk_bytes <= MAX_CHUNK_BYTES and current_token_count >= MIN_CHUNK_TOKENS:
                chunks.append(chunk_text)
            overlap_start = max(0, idx - int(OVERLAP_TOKENS / (sentence_tokens or 1)))
            current_chunk = sentences[overlap_start:idx]
            current_token_count = sum(len(list(NLP(sent))) for sent in current_chunk)
        else:
            current_chunk.append(sentence)
            current_token_count += sentence_tokens

    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunk_bytes = len(chunk_text.encode('utf-8'))
        if chunk_bytes <= MAX_CHUNK_BYTES and current_token_count >= MIN_CHUNK_TOKENS:
            chunks.append(chunk_text)

    return chunks

def process_file(file_path, cache):
    file_hash = get_file_hash(file_path)
    if file_path in cache and cache[file_path] == file_hash:
        logging.info(f"Pulando {file_path} (já processado e inalterado).")
        return []
    logging.info(f"Iniciando processamento de {file_path}.")
    text = extract_text_from_file(file_path)
    if not text:
        logging.warning(f"Nenhum texto extraído de {file_path}.")
        return []
    cleaned_text = preprocess_text(text)
    chunks = semantic_segmentation_with_overlap(cleaned_text)
    if chunks:
        logging.info(f"Processado: {file_path}. Gerados {len(chunks)} chunks.")
        print(f"Processado: {file_path}. Gerados {len(chunks)} chunks.")
    else:
        logging.warning(f"Nenhum chunk válido gerado para {file_path}.")
    cache[file_path] = file_hash
    return chunks

def save_chunks_to_json(chunks, output_json):
    if not chunks:
        print("Nenhum chunk para salvar.")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=4)
        return
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
    print(f"Salvos {len(chunks)} novos chunks em {output_json}.")

if __name__ == "__main__":
    dir_path = "app/rag_data/arquivos"
    output_json = "app/rag_data/chunks.json"
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    if os.path.exists(output_json):
        os.remove(output_json)
        print(f"Arquivo {output_json} removido para nova execução.")

    # Carregar cache
    cache = load_cache()

    # Lista de arquivos a processar recursivamente
    file_extensions = [".pdf", ".docx", ".doc", ".xls", ".xlsx", ".txt"]
    files = []
    for root, _, filenames in os.walk(dir_path):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in file_extensions:
                files.append(os.path.join(root, filename))
    print(f"Encontrados {len(files)} arquivos em pastas aninhadas para processar.")

    # Processamento paralelo com limite de processos
    with Pool(min(cpu_count(), 8)) as pool:  # Limita a 8 processos para evitar sobrecarga
        all_chunks = pool.starmap(process_file, [(f, cache) for f in files])

    # Combinar todos os chunks
    combined_chunks = [chunk for file_chunks in all_chunks for chunk in file_chunks if chunk]

    # Salvar chunks e atualizar cache
    save_chunks_to_json(combined_chunks, output_json)
    save_cache(cache)