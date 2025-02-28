import os
import sys
import timeit
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models import modelo_x_response, modelo_y_response

# Conjunto de testes sintéticos
queries = [
    "Qual é o propósito do artigo 10 da constituição?",
    "O que diz a Lei Federal n.º 8.112/90 sobre férias?",
    "Como funciona o imposto de renda no Brasil?",
    "Quais são os direitos assegurados pelo art. 5º?",
    "O que é a política de cotas raciais?"
]

def test_modelo_x():
    """Testa o Modelo X (sem RAG) apenas por latência."""
    print("### Testando Modelo X (sem RAG) ###")
    total_times = []
    for query in queries:
        start_time = timeit.default_timer()
        modelo_x_response(query, use_cache=False)  # Desativa o cache
        total_time = timeit.default_timer() - start_time
        total_times.append(total_time)
        print(f"Tempo total para '{query}': {total_time:.4f} segundos")
    avg_time = sum(total_times) / len(total_times)
    print(f"Tempo médio do Modelo X: {avg_time:.4f} segundos\n")

def test_modelo_y(top_k):
    """Testa o Modelo Y (com RAG) para um dado top_k apenas por latência."""
    print(f"### Testando Modelo Y com top_k={top_k} ###")
    total_times = []
    for query in queries:
        start_time = timeit.default_timer()
        modelo_y_response(query, top_k=top_k, use_cache=False)  # Desativa o cache
        total_time = timeit.default_timer() - start_time
        total_times.append(total_time)
        print(f"Tempo total para '{query}': {total_time:.4f} segundos")
    avg_time = sum(total_times) / len(total_times)
    print(f"Tempo médio do Modelo Y com top_k={top_k}: {avg_time:.4f} segundos\n")

if __name__ == "__main__":
    test_modelo_x()
    print("=" * 50)
    test_modelo_y(top_k=5)
    print("=" * 50)
    test_modelo_y(top_k=10)

    # Expectativas de qualidade (teóricas)
    print("### Expectativas de Qualidade (Teóricas) ###")
    print("Modelo X (sem RAG):")
    print("- Melhor qualidade esperada: Menor, pois depende apenas do conhecimento interno do Gemini.")
    print("- Pior qualidade esperada: Pode falhar em detalhes específicos dos documentos locais.")
    print("\nModelo Y com top_k=5:")
    print("- Melhor qualidade esperada: Boa, com contexto limitado mas relevante.")
    print("- Pior qualidade esperada: Pode perder detalhes se os 5 chunks forem insuficientes.")
    print("\nModelo Y com top_k=10:")
    print("- Melhor qualidade esperada: Maior, com mais contexto para respostas detalhadas.")
    print("- Pior qualidade esperada: Pode incluir chunks irrelevantes, adicionando ruído.")