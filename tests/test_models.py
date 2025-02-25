import unittest
from unittest.mock import patch, Mock
import numpy as np
import requests  # Import necessário
from app.models import modelo_x_response, modelo_y_response, embed_query, generate_response

class TestModels(unittest.TestCase):

    def setUp(self):
        # Configuração inicial para cada teste
        self.valid_response = {
            "choices": [
                {
                    "message": {
                        "content": "Resposta mockada",
                        "role": "assistant"
                    },
                    "finish_reason": "stop"
                }
            ]
        }
        self.error_response = {
            "error": {
                "message": "Rate limit exceeded: free-models-per-day",
                "code": 429
            }
        }
        self.unexpected_response = {"random": "data"}
        self.timeout_response = {"error": "Request timeout"}  # Dicionário para simular o timeout

    @patch('requests.post')
    def test_modelo_x_response_success(self, mock_post):
        # Teste: Resposta válida da API sem contexto
        mock_response = Mock()
        mock_response.raise_for_status = Mock(return_value=None)
        mock_response.json.return_value = self.valid_response
        mock_post.return_value = mock_response

        result = modelo_x_response("Teste", [])
        self.assertEqual(result, "Resposta mockada")
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_modelo_y_response_success(self, mock_post):
        # Teste: Resposta válida da API com contexto RAG
        mock_response = Mock()
        mock_response.raise_for_status = Mock(return_value=None)
        mock_response.json.return_value = self.valid_response
        mock_post.return_value = mock_response

        with patch('app.models.embed_query', return_value=np.ones((1, 768), dtype='float32')):
            with patch('faiss.IndexFlatL2.search', return_value=(np.array([[0.1]]), np.array([[0]]))):
                result = modelo_y_response("Teste", [])
                self.assertEqual(result, "Resposta mockada")
                mock_post.assert_called_once()

    @patch('requests.post')
    def test_generate_response_rate_limit(self, mock_post):
        # Teste: Erro de limite de taxa
        mock_response = Mock()
        mock_response.raise_for_status = Mock(return_value=None)
        mock_response.json.return_value = self.error_response
        mock_post.return_value = mock_response

        result = generate_response("Teste")
        self.assertEqual(result, "Erro: Rate limit exceeded: free-models-per-day")
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_generate_response_unexpected_format(self, mock_post):
        # Teste: Formato inesperado da resposta
        mock_response = Mock()
        mock_response.raise_for_status = Mock(return_value=None)
        mock_response.json.return_value = self.unexpected_response
        mock_post.return_value = mock_response

        result = generate_response("Teste")
        self.assertEqual(result, "Erro: Resposta da API em formato inesperado.")
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_generate_response_api_failure(self, mock_post):
        # Teste: Falha na requisição à API (ex.: timeout)
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.RequestException("Timeout")
        mock_response.json.return_value = self.timeout_response  # Configurado para retornar o dicionário diretamente
        mock_response.text = '{"error": "Request timeout"}'  # Mantido para consistência
        mock_post.return_value = mock_response

        result = generate_response("Teste")
        print(f"Result returned: '{result}'")  # Para depuração
        self.assertEqual(result, "Erro: Request timeout")  # Deve corresponder ao retorno exato
        mock_post.assert_called_once()

if __name__ == '__main__':
    unittest.main()