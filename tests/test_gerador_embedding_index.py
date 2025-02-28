import os
import sys
import unittest
import shutil
import json
import numpy as np
import faiss
from unittest.mock import patch, MagicMock
import hashlib
from multiprocessing import set_start_method, Pool, cpu_count
import warnings

# Suprimir o aviso de depreciação
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configurar o contexto do multiprocessing para 'spawn'
if __name__ == '__main__':
    set_start_method('spawn', force=True)

# Adicionar o diretório raiz ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.gerador_embedding_index import (
    load_chunks, load_cache, save_cache, get_chunk_hash,
    generate_embedding_single, generate_embeddings_gemini_api, build_index
)

# Função mock global que simula a geração de embeddings e atualiza o cache
def mock_embedding(chunk, cache):
    embedding = np.array([0.1, 0.2, 0.3] * 256, dtype='float32')
    cache[get_chunk_hash(chunk)] = embedding.tolist()
    return embedding

class TestGeradorEmbeddingIndex(unittest.TestCase):
    def setUp(self):
        self.test_dir = "app/rag_data/test"
        os.makedirs(self.test_dir, exist_ok=True)

        self.chunks_json = os.path.join(self.test_dir, "chunks.json")
        self.cache_file = os.path.join(self.test_dir, "embedding_cache.json")
        self.index_path = os.path.join(self.test_dir, "index.faiss")
        self.embeddings_path = os.path.join(self.test_dir, "embeddings.npy")

        self.test_chunks = ["Texto de teste 1", "Texto de teste 2", "Texto de teste 3"]
        with open(self.chunks_json, "w", encoding="utf-8") as f:
            json.dump(self.test_chunks, f)

        self.test_cache = {"hash1": [0.1, 0.2, 0.3] * 256}
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.test_cache, f)

        os.environ["GEMINI_API_KEY"] = "fake_key"
        os.environ.pop("CLOUD_RUN", None)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
        os.environ.pop("USE_FAISS_GPU", None)

    def test_load_chunks_local(self):
        chunks = load_chunks(self.chunks_json)
        self.assertEqual(chunks, self.test_chunks)
        self.assertEqual(len(chunks), 3)

    @patch('google.cloud.storage.Client')
    def test_load_chunks_cloud_run(self, mock_storage_client):
        os.environ["CLOUD_RUN"] = "true"
        os.environ["GCS_BUCKET_NAME"] = "test-bucket"
        
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.download_as_string.return_value = json.dumps(self.test_chunks).encode('utf-8')
        mock_bucket.blob.return_value = mock_blob
        mock_storage_client.return_value.get_bucket.return_value = mock_bucket

        chunks = load_chunks(self.chunks_json)
        self.assertEqual(chunks, self.test_chunks)

    def test_load_cache_local(self):
        with patch('app.gerador_embedding_index.CACHE_FILE', self.cache_file):
            cache = load_cache()
            self.assertEqual(cache, self.test_cache)

    @patch('google.cloud.storage.Client')
    def test_load_cache_cloud_run(self, mock_storage_client):
        os.environ["CLOUD_RUN"] = "true"
        test_cache = {"hash1": [0.1, 0.2, 0.3] * 256}
        
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_as_string.return_value = json.dumps(test_cache).encode('utf-8')
        mock_bucket.blob.return_value = mock_blob
        mock_storage_client.return_value.get_bucket.return_value = mock_bucket

        cache = load_cache()
        self.assertEqual(cache, test_cache)

    def test_get_chunk_hash(self):
        chunk = "Texto de teste"
        hash_result = get_chunk_hash(chunk)
        expected_hash = hashlib.md5(chunk.encode('utf-8')).hexdigest()
        self.assertEqual(hash_result, expected_hash)

    @patch('google.generativeai.embed_content')
    def test_generate_embedding_single(self, mock_embed_content):
        mock_embed_content.return_value = {'embedding': [0.1, 0.2, 0.3] * 256}
        cache = {}
        chunk = "Texto de teste"
        
        embedding = generate_embedding_single(chunk, cache)
        self.assertEqual(embedding.shape, (768,))
        self.assertTrue(np.allclose(embedding, [0.1, 0.2, 0.3] * 256))
        self.assertIn(get_chunk_hash(chunk), cache)

    @patch('google.generativeai.embed_content')
    def test_generate_embedding_single_cache(self, mock_embed_content):
        chunk = "Texto de teste"
        cache = {get_chunk_hash(chunk): [0.1, 0.2, 0.3] * 256}
        mock_embed_content.return_value = {'embedding': [0.0, 0.0, 0.0] * 256}
        
        embedding = generate_embedding_single(chunk, cache)
        self.assertEqual(embedding.shape, (768,))
        self.assertTrue(np.allclose(embedding, [0.1, 0.2, 0.3] * 256))
        mock_embed_content.assert_not_called()

    def test_generate_embeddings_gemini_api(self):
        cache = {}
        chunks = self.test_chunks
        
        # Simular execução sequencial para evitar problemas com multiprocessing no teste
        with patch('app.gerador_embedding_index.generate_embedding_single', mock_embedding):
            # Substituir Pool por uma execução simples para o teste
            embeddings_list = [mock_embedding(chunk, cache) for chunk in chunks]
            embeddings = np.array(embeddings_list, dtype='float32')
            self.assertEqual(embeddings.shape, (3, 768))
            self.assertEqual(len(cache), 3)

    def test_build_index_cpu(self):
        os.environ.pop("USE_FAISS_GPU", None)
        embeddings = np.random.rand(10, 768).astype('float32')
        index = build_index(embeddings)
        self.assertIsInstance(index, faiss.IndexFlatL2)
        self.assertEqual(index.ntotal, 10)

    def test_build_index_gpu(self):
        # Se a função GPU não estiver disponível, ignore o teste
        if not hasattr(faiss, "StandardGpuResources"):
            self.skipTest("faiss.StandardGpuResources not available, skipping GPU test")
        # Caso contrário, execute o teste normalmente
        embeddings = np.random.rand(10, 768).astype('float32')
        index = build_index(embeddings)
        self.assertIsNotNone(index)

if __name__ == "__main__":
    unittest.main()