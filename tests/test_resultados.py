import unittest
from unittest.mock import patch
import pandas as pd
from app.main import app

class TestResultados(unittest.TestCase):

    def setUp(self):
        app.testing = True
        self.client = app.test_client()

    def test_dados_validos(self):
        avaliacoes_df = pd.DataFrame({
            'conversa_id': ['id1', 'id2', 'id3', 'id4', 'id5', 'id6', 'id7', 'id8', 'id9', 'id10'],
            'modelo_vencedor': ['Modelo A', 'Modelo B', 'Modelo A', 'Modelo B', 'Modelo A',
                               'Modelo A', 'Modelo B', 'Modelo A', 'Modelo B', 'Modelo A'],
            'nome': ['user1', 'user2', 'user3', 'user4', 'user5', 'user6', 'user7', 'user8', 'user9', 'user10']
        })
        conversas_df = pd.DataFrame({
            'id': ['id1', 'id2', 'id3', 'id4', 'id5', 'id6', 'id7', 'id8', 'id9', 'id10'],
            'modelo_a': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y'],
            'modelo_b': ['Y', 'X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y', 'X']
        })
        proficiencias_df = pd.DataFrame({
            'conversa_id': ['id1', 'id2', 'id3', 'id4', 'id5', 'id6', 'id7', 'id8', 'id9', 'id10'],
            'nivel': ['Iniciante'] * 5 + ['Avançado'] * 5
        })

        with patch('app.app.pd.read_sql', side_effect=[avaliacoes_df, conversas_df, proficiencias_df]):
            response = self.client.get('/resultados')
            self.assertEqual(response.status_code, 200)
            self.assertIn('Estatísticas Descritivas por Proficiência', response.data.decode('utf-8'))
            self.assertNotIn('Ainda não há dados suficientes', response.data.decode('utf-8'))

    def test_dados_insuficientes_poucos(self):
        avaliacoes_df = pd.DataFrame({
            'conversa_id': ['id1', 'id2', 'id3'],
            'modelo_vencedor': ['Modelo A', 'Modelo B', 'Modelo A'],
            'nome': ['user1', 'user2', 'user3']
        })
        conversas_df = pd.DataFrame({
            'id': ['id1', 'id2', 'id3'],
            'modelo_a': ['X', 'Y', 'X'],
            'modelo_b': ['Y', 'X', 'Y']
        })
        proficiencias_df = pd.DataFrame({
            'conversa_id': ['id1', 'id2', 'id3'],
            'nivel': ['Iniciante', 'Iniciante', 'Avançado']
        })

        with patch('app.app.pd.read_sql', side_effect=[avaliacoes_df, conversas_df, proficiencias_df]):
            response = self.client.get('/resultados')
            self.assertEqual(response.status_code, 200)
            self.assertIn('Ainda não há dados suficientes para afirmar relevância estatística', response.data.decode('utf-8'))

    def test_dados_insuficientes_sem_variacao(self):
        avaliacoes_df = pd.DataFrame({
            'conversa_id': ['id1', 'id2', 'id3', 'id4', 'id5', 'id6', 'id7', 'id8', 'id9', 'id10'],
            'modelo_vencedor': ['Modelo A'] * 10,
            'nome': ['user' + str(i) for i in range(1, 11)]
        })
        conversas_df = pd.DataFrame({
            'id': ['id1', 'id2', 'id3', 'id4', 'id5', 'id6', 'id7', 'id8', 'id9', 'id10'],
            'modelo_a': ['X'] * 10,
            'modelo_b': ['Y'] * 10
        })
        proficiencias_df = pd.DataFrame({
            'conversa_id': ['id1', 'id2', 'id3', 'id4', 'id5', 'id6', 'id7', 'id8', 'id9', 'id10'],
            'nivel': ['Iniciante'] * 5 + ['Avançado'] * 5
        })

        with patch('app.app.pd.read_sql', side_effect=[avaliacoes_df, conversas_df, proficiencias_df]):
            response = self.client.get('/resultados')
            self.assertEqual(response.status_code, 200)
            self.assertIn('Ainda não há dados suficientes para afirmar relevância estatística', response.data.decode('utf-8'))

if __name__ == '__main__':
    unittest.main()