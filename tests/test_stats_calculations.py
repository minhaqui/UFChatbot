import unittest
import pandas as pd
from app.stats import calculate_statistics, anonimizar_nome

class TestStatsCalculations(unittest.TestCase):
    def test_calculate_statistics_insufficient_data(self):
        # Dados insuficientes: menos de 10 linhas
        avaliacoes_df = pd.DataFrame({
            'conversa_id': ['id1', 'id2', 'id3'],
            'modelo_vencedor': ['Chat A', 'Chat B', 'Chat A'],
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
        stats_dict = calculate_statistics(avaliacoes_df, conversas_df, proficiencias_df)
        self.assertIn('mensagem', stats_dict)
        self.assertEqual(stats_dict['mensagem'], "Ainda não há dados suficientes para afirmar relevância estatística completa.")

    def test_calculate_statistics_sufficient_data(self):
        # 10 linhas com resultados alternados
        avaliacoes_df = pd.DataFrame({
            'conversa_id': [f'id{i}' for i in range(1, 11)],
            'modelo_vencedor': ['Chat A', 'Chat B'] * 5,
            'nome': [f'user{i}' for i in range(1, 11)]
        })
        # Para "Chat A" usamos 'X' e para "Chat B" usamos 'Y'
        conversas_df = pd.DataFrame({
            'id': [f'id{i}' for i in range(1, 11)],
            'modelo_a': ['X'] * 5 + ['Y'] * 5,
            'modelo_b': ['Y'] * 5 + ['X'] * 5,
        })
        proficiencias_df = pd.DataFrame({
            'conversa_id': [f'id{i}' for i in range(1, 11)],
            'nivel': ['Iniciante'] * 10
        })
        stats_dict = calculate_statistics(avaliacoes_df, conversas_df, proficiencias_df)
        # Conforme o mapeamento, espera-se p̂ = 0.4
        self.assertIn('p_hat', stats_dict)
        self.assertAlmostEqual(stats_dict['p_hat'], 0.4, places=3)
        self.assertIn('p_valor', stats_dict)
        self.assertIn('margem_erro', stats_dict)
        self.assertIn('ic_inferior', stats_dict)
        self.assertIn('ic_superior', stats_dict)

    def test_calculate_statistics_with_multiple_levels(self):
        # Teste com 20 avaliações divididas em dois níveis de proficiência
        avaliacoes_df = pd.DataFrame({
            'conversa_id': [f'id{i}' for i in range(1, 21)],
            'modelo_vencedor': ['Chat A', 'Chat B'] * 10,
            'nome': [f'user{i}' for i in range(1, 21)]
        })
        conversas_df = pd.DataFrame({
            'id': [f'id{i}' for i in range(1, 21)],
            'modelo_a': ['X'] * 10 + ['Y'] * 10,
            'modelo_b': ['Y'] * 10 + ['X'] * 10,
        })
        proficiencias_df = pd.DataFrame({
            'conversa_id': [f'id{i}' for i in range(1, 21)],
            'nivel': ['Iniciante'] * 10 + ['Avançado'] * 10
        })
        stats_dict = calculate_statistics(avaliacoes_df, conversas_df, proficiencias_df)
        # Verifica que as tabelas foram geradas e que, se houver dados suficientes, os cálculos de hipótese estão presentes
        self.assertIn('desc_stats', stats_dict)
        self.assertIn('tabela_avaliacoes', stats_dict)
        if 'mensagem' not in stats_dict:
            self.assertIn('p_hat', stats_dict)
            self.assertIn('p_valor', stats_dict)
            self.assertIn('margem_erro', stats_dict)
            self.assertIn('ic_inferior', stats_dict)
            self.assertIn('ic_superior', stats_dict)

if __name__ == '__main__':
    unittest.main()
