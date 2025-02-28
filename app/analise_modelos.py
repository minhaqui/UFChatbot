import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import logit
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import hashlib

# Configurar Matplotlib para usar a fonte Roboto
plt.rcParams['font.family'] = 'Roboto'

# Conectar ao banco de dados SQLite
engine = create_engine('sqlite:///chat.db')

# Carregar os dados
avaliacoes_df = pd.read_sql('SELECT * FROM avaliacao', engine)
conversas_df = pd.read_sql('SELECT * FROM conversa', engine)
proficiencias_df = pd.read_sql('SELECT * FROM proficiencia', engine)

# Função para anonimizar nomes
def anonimizar_nome(nome):
    if pd.isna(nome):
        return 'Anônimo'
    return hashlib.sha256(nome.encode()).hexdigest()[:10]

# Aplicar anonimização
avaliacoes_df['nome'] = avaliacoes_df['nome'].apply(anonimizar_nome)

# Juntar os DataFrames
df = avaliacoes_df.merge(conversas_df, left_on='conversa_id', right_on='id', how='left') \
                 .merge(proficiencias_df, on='conversa_id', how='left')

# Criar variável binária: 1 se Modelo Y (com RAG) foi preferido, 0 caso contrário
df['prefere_rag'] = np.where(
    (df['modelo_vencedor'] == 'Modelo A') & (df['modelo_a'] == 'Y') |
    (df['modelo_vencedor'] == 'Modelo B') & (df['modelo_b'] == 'Y'), 1, 0
)

# Bloco 1: Pré-processamento e Análise Exploratória
print("=== Bloco 1: Pré-processamento e Análise Exploratória ===")

# Agrupar por nível de proficiência
grupos = df.groupby('nivel')
desc_stats = grupos['prefere_rag'].agg(['mean', 'std', 'count']).rename(columns={
    'mean': 'Média', 'std': 'Desvio Padrão', 'count': 'Contagem'
})
print("\nEstatísticas Descritivas por Nível de Proficiência:")
print(desc_stats)

# Gráfico de barras
plt.figure(figsize=(10, 6))
sns.barplot(x=desc_stats.index, y=desc_stats['Média'], palette='viridis')
plt.title('Proporção de Preferência pelo Modelo Y (RAG) por Proficiência')
plt.xlabel('Nível de Proficiência')
plt.ylabel('Proporção de Preferência')
plt.ylim(0, 1)
plt.savefig('exploratoria.png')
plt.show()

# Comentário: Histogramas ou boxplots não são ideais para dados binários; usamos barras.

# Bloco 2: Testes de Hipótese Nula com Intervalo de Confiança e Margem de Erro
print("\n=== Bloco 2: Teste de Hipótese Nula ===")

def teste_binomial_e_ic(dados, nivel_confianca=0.95):
    """Realiza teste binomial e calcula intervalo de confiança."""
    n = len(dados)
    sucessos = dados.sum()
    p_hat = sucessos / n
    p_valor = stats.binom_test(sucessos, n, p=0.5, alternative='two-sided')
    z = stats.norm.ppf(1 - (1 - nivel_confianca) / 2)
    erro_padrao = np.sqrt(p_hat * (1 - p_hat) / n)
    margem_erro = z * erro_padrao
    ic_inferior = max(0, p_hat - margem_erro)
    ic_superior = min(1, p_hat + margem_erro)
    return p_hat, p_valor, ic_inferior, ic_superior, margem_erro

# Teste no conjunto completo
p_hat, p_valor, ic_inferior, ic_superior, margem_erro = teste_binomial_e_ic(df['prefere_rag'])
print(f"Proporção Estimada: {p_hat:.3f}")
print(f"P-valor: {p_valor:.4f}")
print(f"Intervalo de Confiança 95%: [{ic_inferior:.3f}, {ic_superior:.3f}]")
print(f"Margem de Erro: ±{margem_erro:.3f}")

# Bloco 3: Análise Estratificada
print("\n=== Bloco 3: Análise Estratificada ===")

resultados_estratos = []
for nivel, grupo in grupos:
    p_hat, p_valor, ic_inferior, ic_superior, margem_erro = teste_binomial_e_ic(grupo['prefere_rag'])
    print(f"\nNível: {nivel}")
    print(f"Proporção: {p_hat:.3f}, P-valor: {p_valor:.4f}")
    print(f"IC 95%: [{ic_inferior:.3f}, {ic_superior:.3f}], Margem: ±{margem_erro:.3f}")
    resultados_estratos.append({
        'Nível': nivel, 'Proporção': p_hat, 'IC_Inferior': ic_inferior, 'IC_Superior': ic_superior
    })

# Gráfico dos estratos
resultados_df = pd.DataFrame(resultados_estratos)
plt.figure(figsize=(10, 6))
sns.barplot(x='Nível', y='Proporção', data=resultados_df, palette='viridis')
plt.errorbar(x=resultados_df.index, y=resultados_df['Proporção'],
             yerr=[resultados_df['Proporção'] - resultados_df['IC_Inferior'],
                   resultados_df['IC_Superior'] - resultados_df['Proporção']],
             fmt='none', c='black', capsize=5)
plt.title('Preferência pelo Modelo Y por Nível de Proficiência com IC 95%')
plt.xlabel('Nível de Proficiência')
plt.ylabel('Proporção')
plt.ylim(0, 1)
plt.savefig('estratificada.png')
plt.show()

# Bloco 4: Modelagem com Interação
print("\n=== Bloco 4: Modelagem com Interação ===")

# Codificar proficiência como dummies
df = pd.get_dummies(df, columns=['nivel'], drop_first=True)

# Variável indicadora: Modelo A é Y
df['modelo_a_Y'] = (df['modelo_a'] == 'Y').astype(int)

# Fórmula com interações
termos_interacao = [col for col in df.columns if col.startswith('nivel_')]
formula = 'prefere_rag ~ modelo_a_Y + ' + ' + '.join(termos_interacao) + \
          ' + modelo_a_Y:(' + ' + '.join(termos_interacao) + ')'

# Ajustar modelo
modelo = logit(formula, data=df).fit()
print(modelo.summary())

# Salvar resultados
plt.savefig('modelagem.png')