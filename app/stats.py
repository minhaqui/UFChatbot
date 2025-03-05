"""
File: stats.py
Description: Este módulo contém funções para calcular estatísticas descritivas e testes de hipótese para o projeto de comparação de LLMs.
As análises são agrupadas por níveis de proficiência dos usuários e incluem contagem de vitórias dos modelos X e Y.
Author: Bruno Cipriano Minhaqui da Silva
Date: Fevereiro de 2025
"""

import pandas as pd
import numpy as np
from scipy import stats
import math
import logging

logging.basicConfig(level=logging.ERROR)

FALLBACK_MSG = "Sem dados suficientes para análise."

def anonimizar_nome(nome):
    if pd.isna(nome) or len(nome.strip()) < 2:
        return 'Anônimo'
    nome = nome.strip()
    return nome[0] + '*' * (len(nome) - 2) + nome[-1]

def calculate_statistics(avaliacoes_df, conversas_df, proficiencias_df):
    # Renomeia as colunas de conversas_df se necessário
    if 'chat_a' not in conversas_df.columns and 'modelo_a' in conversas_df.columns:
        conversas_df = conversas_df.rename(columns={'modelo_a': 'chat_a', 'modelo_b': 'chat_b'})
    
    # Mescla os dataframes
    df = avaliacoes_df.merge(conversas_df, left_on='conversa_id', right_on='id', how='left')
    df = df.merge(proficiencias_df, on='conversa_id', how='left')
    
    if 'data_hora' in df.columns:
        df['data_hora'] = pd.to_datetime(df['data_hora'])
        df = df.sort_values(by='data_hora', ascending=False)
    
    # Mapeia o modelo vencedor: se "Chat A" então usa chat_a, senão chat_b
    df['modelo_vencedor'] = df.apply(lambda row: row['chat_a'] if row['modelo_vencedor'] == 'Chat A' else row['chat_b'], axis=1)
    
    # Define as colunas necessárias para tabela_avaliacoes
    if 'data_hora' in df.columns:
        required_columns = ['data_hora', 'modelo_vencedor', 'nome', 'nivel']
    else:
        required_columns = ['conversa_id', 'modelo_vencedor', 'nome', 'nivel']
    
    # Seleciona apenas as colunas desejadas
    tabela_avaliacoes = df[required_columns].copy()
    
    # Renomeia as colunas (exatamente 4 colunas)
    tabela_avaliacoes.columns = ['Data/Hora da Avaliação', 'Modelo Vencedor', 'Nome', 'Nível de Proficiência']
    
    # Anonimiza o nome e adiciona como nova coluna
    tabela_avaliacoes['Nome (Anonimizado)'] = tabela_avaliacoes['Nome'].apply(anonimizar_nome)
    
    # Remove a coluna 'Nome' original
    tabela_avaliacoes = tabela_avaliacoes.drop(columns=['Nome'])
    
    # Mapeia o Modelo Vencedor
    tabela_avaliacoes['Modelo Vencedor'] = tabela_avaliacoes['Modelo Vencedor'].map({'X': 'Modelo X', 'Y': 'Modelo Y'})
    
    # Limita a 20 linhas
    tabela_avaliacoes = tabela_avaliacoes.head(20)
    
    # Estatísticas descritivas: calcular contagem de vitórias para os modelos Y e X, por nível
    desc_stats = df.groupby('nivel')['modelo_vencedor'].agg(
        vit_y=lambda x: (x == 'Y').sum(),
        vit_x=lambda x: (x == 'X').sum()
    ).reset_index()
    desc_stats = desc_stats.rename(columns={'vit_x': 'Vitórias do Modelo X', 'vit_y': 'Vitórias do Modelo Y'})
    niveis = ['Iniciante', 'Básico', 'Intermediário', 'Avançado', 'Especialista']
    desc_stats = desc_stats.set_index('nivel').reindex(niveis, fill_value=0).reset_index()
    desc_stats = desc_stats.rename(columns={'nivel': 'Nível de Proficiência'})
    
    # Calcular totais diretamente das colunas
    total_vitorias_x = desc_stats['Vitórias do Modelo X'].sum()
    total_vitorias_y = desc_stats['Vitórias do Modelo Y'].sum()
    
    # Criar a linha total
    total_row = pd.DataFrame([{
        'Nível de Proficiência': 'Total',
        'Vitórias do Modelo X': total_vitorias_x,
        'Vitórias do Modelo Y': total_vitorias_y
    }])
    
    # Concatenar ao desc_stats
    desc_stats = pd.concat([desc_stats, total_row], ignore_index=True)
    
    # Adicionar a coluna "Y/X"
    desc_stats['Y/X'] = desc_stats.apply(
        lambda row: row['Vitórias do Modelo Y'] / row['Vitórias do Modelo X'] if row['Vitórias do Modelo X'] != 0 else 'N/A',
        axis=1
    )
    
    stats_dict = {}
    n_overall = len(df)
    if n_overall > 0 and df['modelo_vencedor'].nunique() >= 2:
        overall_p_hat = (df['modelo_vencedor'] == 'Y').mean()
        stats_dict['mensagem'] = ""
        stats_dict['p_hat'] = round(overall_p_hat, 3)
        se_null = math.sqrt(0.5 * 0.5 / n_overall)
        overall_z = (overall_p_hat - 0.5) / se_null
        overall_p_valor = 2 * (1 - stats.norm.cdf(abs(overall_z)))
        stats_dict['p_valor'] = round(overall_p_valor, 3)
        se_obs = math.sqrt(overall_p_hat * (1 - overall_p_hat) / n_overall)
        overall_margin = 1.96 * se_obs
        stats_dict['margem_erro'] = round(overall_margin, 3)
        stats_dict['ic_inferior'] = round(max(0, overall_p_hat - overall_margin), 3)
        stats_dict['ic_superior'] = round(min(1, overall_p_hat + overall_margin), 3)
    else:
        stats_dict['mensagem'] = FALLBACK_MSG
        stats_dict['p_hat'] = FALLBACK_MSG
        stats_dict['p_valor'] = FALLBACK_MSG
        stats_dict['margem_erro'] = FALLBACK_MSG
        stats_dict['ic_inferior'] = FALLBACK_MSG
        stats_dict['ic_superior'] = FALLBACK_MSG
    
    overall_test = {
        "Segmento": "Total",
        "Tamanho da Amostra": n_overall,
        "Proporção Amostral p-hat": stats_dict['p_hat'],
        "Valor-p": stats_dict['p_valor'],
        "Margem de Erro": stats_dict['margem_erro'],
        "Intervalo de Confiança": (f"[{stats_dict['ic_inferior']}, {stats_dict['ic_superior']}]" 
                                   if stats_dict['ic_inferior'] != FALLBACK_MSG 
                                   else FALLBACK_MSG),
        "Significativo?": ("Sim" if isinstance(stats_dict['p_valor'], (int, float)) and stats_dict['p_valor'] < 0.05 
                           else ("Não" if isinstance(stats_dict['p_valor'], (int, float)) else FALLBACK_MSG))
    }
    
    group_tests = []
    for nivel in niveis:
        group = df[df['nivel'] == nivel]
        n_level = len(group)
        if n_level > 0 and group['modelo_vencedor'].nunique() >= 2:
            p_hat = (group['modelo_vencedor'] == 'Y').mean()
            se_null = math.sqrt(0.5 * 0.5 / n_level)
            z = (p_hat - 0.5) / se_null
            p_valor = 2 * (1 - stats.norm.cdf(abs(z)))
            se_obs = math.sqrt(p_hat * (1 - p_hat) / n_level)
            margin = 1.96 * se_obs
            ci = f"[{round(max(0, p_hat - margin), 3)}, {round(min(1, p_hat + margin), 3)}]"
            significant = "Sim" if p_valor < 0.05 else "Não"
            group_tests.append({
                "Segmento": nivel,
                "Tamanho da Amostra": n_level,
                "Proporção Amostral p-hat": round(p_hat, 3),
                "Valor-p": round(p_valor, 3),
                "Margem de Erro": round(margin, 3),
                "Intervalo de Confiança": ci,
                "Significativo?": significant
            })
        else:
            group_tests.append({
                "Segmento": nivel,
                "Tamanho da Amostra": n_level,
                "Proporção Amostral p-hat": FALLBACK_MSG,
                "Valor-p": FALLBACK_MSG,
                "Margem de Erro": FALLBACK_MSG,
                "Intervalo de Confiança": FALLBACK_MSG,
                "Significativo?": FALLBACK_MSG
            })
    
    # Criar teste_hipotese_df com group_tests primeiro e overall_test no final
    teste_hipotese_df = pd.DataFrame(group_tests + [overall_test])
    
    stats_dict.update({
        'desc_stats': desc_stats.to_html(index=False, classes='dataframe'),
        'tabela_avaliacoes': tabela_avaliacoes.to_html(index=False, classes='dataframe'),
        'teste_hipotese': teste_hipotese_df.to_html(index=False, classes='dataframe')
    })
    
    return stats_dict