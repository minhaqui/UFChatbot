import os
import sys
import unittest
import shutil
from multiprocessing import Pool, cpu_count
import spacy
import warnings  # Adicionado para suprimir aviso

# Adicionar o diretório raiz ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.extrair_texto import (
    extract_text_from_pdf, extract_text_from_docx, extract_text_from_doc,
    extract_text_from_excel, extract_text_from_txt, extract_text_from_file,
    preprocess_text, semantic_segmentation_with_overlap, process_file,
    load_cache, save_cache
)
from docx import Document
import pandas as pd
import PyPDF2
import time

# Suprimir ResourceWarning do openpyxl
warnings.filterwarnings("ignore", category=ResourceWarning)

# Instanciar NLP para os testes
NLP = spacy.load("pt_core_news_sm", disable=["ner", "parser"])
NLP.add_pipe('sentencizer')

class TestExtrairTexto(unittest.TestCase):
    def setUp(self):
        self.test_dir = "app/rag_data/arquivos_teste"
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "subpasta1"), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "subpasta2"), exist_ok=True)

        self.files = {
            "pdf": os.path.join(self.test_dir, "teste.pdf"),
            "docx": os.path.join(self.test_dir, "teste.docx"),
            "doc": os.path.join(self.test_dir, "teste.doc"),
            "xls": os.path.join(self.test_dir, "teste.xls"),
            "xlsx": os.path.join(self.test_dir, "subpasta1/teste.xlsx"),
            "txt": os.path.join(self.test_dir, "subpasta2/teste.txt")
        }

        # Criar PDF (vazio por padrão)
        with open(self.files["pdf"], "wb") as f:
            pdf_writer = PyPDF2.PdfWriter()
            page = pdf_writer.add_blank_page(width=612, height=792)
            pdf_writer.write(f)

        # Criar DOCX
        doc = Document()
        doc.add_paragraph("Texto de teste no DOCX com conteúdo suficiente para chunking.")
        doc.save(self.files["docx"])

        # Criar TXT com mais texto
        with open(self.files["txt"], "w", encoding="utf-8") as f:
            f.write("Texto de teste no TXT. " * 50)

        # Criar XLSX
        df = pd.DataFrame({"Coluna1": ["Texto"] * 10, "Coluna2": ["de teste"] * 10})
        with pd.ExcelWriter(self.files["xlsx"], engine='openpyxl') as writer:
            df.to_excel(writer, index=False)

        # Simular DOC e XLS
        with open(self.files["doc"], "w", encoding="utf-8") as f:
            f.write("Simulação de texto no DOC")
        with open(self.files["xls"], "w", encoding="utf-8") as f:
            f.write("Simulação de texto no XLS")

        self.cache = load_cache()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
        if os.path.exists("app/rag_data/chunk_cache.json"):
            os.remove("app/rag_data/chunk_cache.json")

    def test_extract_text_from_pdf(self):
        text = extract_text_from_pdf(self.files["pdf"])
        self.assertTrue(isinstance(text, str))

    def test_extract_text_from_docx(self):
        text = extract_text_from_docx(self.files["docx"])
        self.assertTrue(isinstance(text, str))
        self.assertIn("Texto de teste no DOCX", text)

    def test_extract_text_from_doc(self):
        text = extract_text_from_doc(self.files["doc"])
        self.assertTrue(isinstance(text, str))

    def test_extract_text_from_excel(self):
        text = extract_text_from_excel(self.files["xlsx"])
        self.assertTrue(isinstance(text, str))
        self.assertIn("Texto de teste", text)

    def test_extract_text_from_txt(self):
        text = extract_text_from_txt(self.files["txt"])
        self.assertTrue(isinstance(text, str))
        self.assertIn("Texto de teste no TXT", text)

    def test_extract_text_from_file(self):
        for file_type, path in self.files.items():
            text = extract_text_from_file(path)
            self.assertTrue(isinstance(text, str))
            if file_type not in ["pdf", "doc", "xls"]:
                self.assertTrue(len(text) > 0, f"Texto vazio para {file_type}")

        invalid_path = os.path.join(self.test_dir, "invalido.xyz")
        with open(invalid_path, "w") as f:
            f.write("Algo")
        text = extract_text_from_file(invalid_path)
        self.assertEqual(text, "")

    def test_preprocess_text(self):
        sample_text = "Página 1 de 10\nTexto útil\n\n123\nOutro texto"
        cleaned = preprocess_text(sample_text)
        self.assertNotIn("Página", cleaned)
        self.assertNotIn("123", cleaned)
        self.assertIn("Texto útil", cleaned)

    def test_semantic_segmentation_with_overlap(self):
        sample_text = " ".join(["Texto útil para teste."] * 50)
        chunks = semantic_segmentation_with_overlap(sample_text)
        self.assertTrue(len(chunks) > 0, f"Nenhum chunk gerado: {sample_text}")
        for chunk in chunks:
            token_count = len(list(NLP(chunk)))
            byte_count = len(chunk.encode('utf-8'))
            self.assertTrue(100 <= token_count <= 800, f"Token count: {token_count}")
            self.assertTrue(byte_count <= 9000, f"Byte count: {byte_count}")

    def test_process_file(self):
        chunks = process_file(self.files["txt"], self.cache)
        print(f"Chunks gerados para TXT: {chunks}")
        self.assertTrue(len(chunks) > 0, "Nenhum chunk gerado para TXT")
        self.assertIn("Texto de teste no TXT", " ".join(chunks))

    def test_recursive_file_processing(self):
        files = []
        file_extensions = [".pdf", ".docx", ".doc", ".xls", ".xlsx", ".txt"]
        for root, _, filenames in os.walk(self.test_dir):
            for filename in filenames:
                if os.path.splitext(filename)[1].lower() in file_extensions:
                    files.append(os.path.join(root, filename))
        self.assertEqual(len(files), 6)

        start_time = time.time()
        with Pool(min(cpu_count(), 8)) as pool:
            all_chunks = pool.starmap(process_file, [(f, self.cache) for f in files])
        duration = time.time() - start_time
        print(f"Tempo de processamento para 6 arquivos: {duration:.2f} segundos")

        combined_chunks = [chunk for file_chunks in all_chunks for chunk in file_chunks if chunk]
        print(f"Combined chunks: {combined_chunks}")
        self.assertTrue(len(combined_chunks) > 0, "Nenhum chunk combinado gerado")
        combined_text = " ".join(combined_chunks)
        self.assertIn("Texto de teste no TXT", combined_text)

    def test_cache_mechanism(self):
        chunks1 = process_file(self.files["txt"], self.cache)
        print(f"Primeira execução - Chunks: {chunks1}")
        save_cache(self.cache)
        self.assertTrue(len(chunks1) > 0, "Nenhum chunk na primeira execução")

        chunks2 = process_file(self.files["txt"], self.cache)
        print(f"Segunda execução - Chunks: {chunks2}")
        self.assertEqual(len(chunks2), 0, "Cache não funcionou")

if __name__ == "__main__":
    unittest.main()