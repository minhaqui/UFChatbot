# Chat Comparativo de Modelos de IA para Contratações Públicas

## Visão Geral

Este projeto é um protótipo desenvolvido como parte de uma pesquisa de mestrado para comparar o desempenho de dois modelos de inteligência artificial (Modelo X e Modelo Y) no contexto de contratações públicas no Brasil. Inspirado no formato do LMArena, o sistema apresenta uma interface web onde usuários enviam mensagens que são respondidas por duas LLMs, exibidas em chats lado a lado (Modelo A à esquerda e Modelo B à direita). A alocação dos modelos X e Y aos chats A e B é feita aleatoriamente a cada nova conversa, garantindo um teste cego. Os usuários podem avaliar qual modelo fornece as melhores respostas, e os dados coletados são usados para análises estatísticas que avaliam a eficácia de cada modelo.

O Modelo X é implementado diretamente com a API Gemini, fornecendo respostas sem contexto adicional, enquanto o Modelo Y utiliza a mesma API Gemini com acesso a um sistema de Recuperação Aumentada de Geração (RAG) baseado em um corpus de aproximadamente 50 MB (~5000 páginas) de livros e artigos sobre contratações públicas no Brasil. O objetivo é determinar qual abordagem (com ou sem RAG) é mais eficiente para esse domínio específico.

A aplicação é projetada para baixa latência, suporta entre 20 e 2000 usuários em uma semana, e inclui três páginas principais: Chat, Resultados e Sobre. Ela preserva o histórico de conversas, registra informações do usuário (opcionalmente), e realiza análises estatísticas detalhadas das avaliações.

---

## Objetivos

- **Comparação Cega**: Avaliar o desempenho do Modelo X (Gemini puro) e do Modelo Y (Gemini com RAG) em um teste cego, onde os usuários não sabem qual modelo está por trás de cada chat.
- **Contexto Específico**: Configurar as LLMs para responder a perguntas sobre contratações públicas no Brasil, usando prompts iniciais e, para o Modelo Y, um RAG.
- **Interatividade**: Fornecer uma interface web intuitiva com chats duplos, botões de avaliação e registro de proficiência.
- **Coleta e Análise de Dados**: Salvar mensagens, avaliações e informações do usuário em um banco de dados, realizando estatísticas descritivas, cálculo de intervalo de confiança e teste de hipótese.

---

## Estrutura do Projeto

```
PROJETO\
├── .env                      # Contém a SECRET_KEY; outras chaves como variáveis de ambiente no sistema
├── .gcloudignore            # Ignora arquivos no deploy para Google Cloud
├── Dockerfile               # Configuração do container Docker
├── chunking_debug.log       # Log de depuração do processo de chunking
├── README.md                # Documentação do projeto (este arquivo)
├── requirements.txt         # Dependências do Python
├── verificar_chunks.py      # Script para verificar os chunks gerados
├── app/                     # Diretório principal da aplicação Flask
│   ├── __init__.py          # Inicialização do módulo Flask
│   ├── app.py               # Aplicação Flask principal
│   ├── db.py                # Configuração e modelos do banco de dados
│   ├── models.py            # Lógica dos modelos X e Y
│   ├── chat.db              # Banco de dados SQLite
│   ├── stats.py             # Funções para análises estatísticas
│   ├── static/              # Arquivos estáticos
│   │   ├── css/
│   │   │   └── style.css    # Estilização da interface
│   │   └── js/
│   │       └── script.js    # Scripts JavaScript para interatividade
│   │       └── favicon.ico  # Ícone do site
│   ├── templates/           # Templates HTML
│   │   ├── chat.html        # Página principal do chat
│   │   ├── resultados.html  # Página de resultados estatísticos
│   │   └── sobre.html       # Página "Sobre"
│   └── rag_data/            # Dados do RAG
│       ├── chunks.json      # Chunks de texto processados
│       ├── embeddings.npy   # Embeddings NumPy
│       ├── index.faiss      # Índice FAISS
│       └── arquivos/        # Arquivos de diversas fontes para treinar o RAG
└── test/                    # Diretório de testes
    ├── __init__.py          # Inicialização do módulo de testes
    ├── test_app.py          # Testes da aplicação Flask
    ├── test_extrair_texto_PDF.py  # Testes de extração de texto
    ├── test_gerador_embedding_index.py  # Testes de geração de embeddings
    ├── test_models.py       # Testes dos modelos
    ├── test_rag_integration.py  # Testes de integração do RAG
    └── test_stats.py        # Testes das análises estatísticas
```

---

## Lógica do Projeto

### Componentes Principais

1. **Frontend**:
   - **Chat (`chat.html`)**: Interface com dois painéis de chat (Modelo A e B), um campo de mensagem compartilhado, botões de avaliação, campos de identificação opcional (nome, email), e um seletor de proficiência. Inclui instruções no topo e uma animação de confetti ao avaliar.
   - **Resultados (`resultados.html`)**: Exibe análises estatísticas (proporção de vitórias, intervalo de confiança, valor-p) e uma tabela paginada com até 20 linhas de avaliações anonimizadas.
   - **Sobre (`sobre.html`)**: Informações do projeto, ferramentas usadas, contato do autor e agradecimentos.
   - **JavaScript (`script.js`)**: Gerencia eventos de clique, enviando requisições AJAX ao backend e atualizando a interface dinamicamente.

2. **Backend**:
   - **Flask (`app.py`)**: Define rotas para renderizar páginas, processar mensagens, avaliações e proficiências, e integrar com o banco de dados e modelos.
   - **Modelos (`models.py`)**: Contém `modelo_x_response` (Gemini puro) e `modelo_y_response` (Gemini com RAG via `recuperacao.py`).
   - **Banco de Dados (`db.py`)**: Usa SQLite para salvar conversas, mensagens, avaliações e proficiências.
   - **Estatísticas (`stats.py`)**: Calcula análises estatísticas (descritivas, intervalo de confiança, teste de hipótese).

3. **RAG (`recuperacao.py`)**:
   - Usa um índice FAISS (`index.faiss`) e embeddings pré-gerados (`embeddings.npy`) para buscar chunks relevantes de `chunks.json`, fornecendo contexto ao Modelo Y.

### Fluxo de Funcionamento

1. **Início da Conversa**:
   - O usuário acessa `/`, gerando um `session_id` único e uma alocação aleatória de Modelo X e Y para Modelo A e B (`app.py`).
   - O `chat.html` é renderizado com instruções e o `session_id`.

2. **Envio de Mensagem**:
   - O usuário digita uma mensagem e clica em "Enviar".
   - O `script.js` valida a mensagem e envia um POST para `/send_message`.
   - O `app.py` chama `modelo_x_response` e `modelo_y_response`, salva as respostas no banco (`db.py`) e retorna um JSON com as respostas.
   - O frontend atualiza os chats com as respostas, preservando o histórico.

3. **Avaliação**:
   - O usuário clica em "Modelo A é Melhor" ou "Modelo B é Melhor".
   - O `script.js` envia um POST para `/evaluate` com o vencedor, nome e email opcionais.
   - O backend registra a avaliação e retorna uma confirmação, ativando o confetti no frontend.

4. **Proficiência**:
   - O usuário seleciona um nível e clica em "Enviar Proficiência".
   - O `script.js` envia um POST para `/submit_proficiency`, e o backend registra no banco.

5. **Resultados**:
   - A rota `/resultados` exibe estatísticas e uma tabela paginada com avaliações recentes.

---

## Arquitetura

### Frontend
- **HTML/CSS/JavaScript**: `chat.html`, `resultados.html` e `sobre.html` são estilizados com `style.css`. O `script.js` usa Fetch API para comunicação assíncrona, com `canvas-confetti` para animações de vitória.
- **Interatividade**: Os chats são atualizados dinamicamente, e a interface é responsiva para até 2000 usuários.

### Backend
- **Flask**: Framework leve que gerencia rotas, sessões e integração com o banco de dados.
- **SQLAlchemy**: ORM para o banco SQLite (`chat.db`), com tabelas para conversas, mensagens, avaliações e proficiências.
- **Sessão**: O módulo `session` do Flask rastreia o `session_id` e a alocação dos modelos.

### Modelos de IA
- **Modelo X**: Usa a API Gemini diretamente, enviando prompts com a mensagem atual e o histórico da conversa.
- **Modelo Y**: Integra RAG, buscando chunks relevantes com `recuperacao.py` e passando o contexto ao Gemini.
- **Cache**: Dicionários (`cache_x`, `cache_y`) em `models.py` armazenam respostas para reduzir latência.

### RAG
- **FAISS**: Índice vetorial (`index.faiss`) pré-gerado com embeddings de 4385 chunks (~50 MB, ~5000 páginas).
- **Embeddings**: Gerados pela API Gemini (`models/embedding-001`) e salvos em `embeddings.npy`.
- **Chunks**: Extraídos de 26 arquivos em `rag_data/arquivos/` e armazenados em `chunks.json`.

### Banco de Dados
- **SQLite**: Escolhido por simplicidade e adequação para 20-2000 usuários em uma semana.
- **Tabelas**:
  - `conversa`: ID da sessão, alocação dos modelos (A e B).
  - `mensagem_x`, `mensagem_y`: Histórico de mensagens por modelo.
  - `avaliacao`: Vencedor, nome, email.
  - `proficiencia`: Nível de proficiência.

### Testes
- **Unitários**: Testam funções individuais (`test_models.py`, `test_stats.py`).
- **Integração**: Verificam interações entre módulos (`test_rag_integration.py`).
- **Sistema**: Testam o fluxo completo (`test_app.py`).

---

## Pré-requisitos

- **Python 3.8+**
- **Dependências**: Listadas em `requirements.txt` (instale com `pip install -r requirements.txt`):
  - Flask, Flask-SQLAlchemy, google-generativeai, pandas, faiss-cpu, numpy, requests, etc.
- **Chaves de API**:
  - `GEMINI_API_KEY`: Para embeddings e respostas via Gemini.
  - `OPENROUTE_API_KEY`: Para respostas via OpenRouter.

---

## Como Executar

1. **Clone o Repositório**:
   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd PROJETO
   ```

2. **Configure as Variáveis de Ambiente**:
   - No Windows, defina no sistema:
     ```cmd
     set GEMINI_API_KEY=sua_chave_gemini
     set OPENROUTE_API_KEY=sua_chave_openroute
     ```
   - Adicione ao `.env`:
     ```
     SECRET_KEY=sua_chave_secreta
     ```

3. **Instale Dependências**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute a Aplicação**:
   ```bash
   python -m app.app
   ```
   Acesse `http://127.0.0.1:5000/` no navegador.

5. **Interaja**:
   - Envie mensagens na página Chat.
   - Avalie os modelos e registre sua proficiência.
   - Veja os resultados em `/resultados`.

---

## Testes

1. **Unitários**:
   ```bash
   python -m unittest test.test_models
   python -m unittest test.test_stats
   ```

2. **Integração**:
   ```bash
   python -m unittest test.test_rag_integration
   ```

3. **Sistema**:
   ```bash
   python -m unittest test.test_app
   ```

---

## Limitações e Melhorias Futuras

- **Warnings do gRPC**: Logs residuais do gRPC podem aparecer, mas não afetam a funcionalidade.
- **Estatísticas**: Atualmente, `/resultados` mostra apenas uma tabela; implementar `stats.py` com cálculos completos é necessário.
- **Escalabilidade**: Para mais de 2000 usuários, considere um banco como PostgreSQL e um servidor WSGI (Gunicorn).

---

## Autor

- **Nome:** Bruno Cipriano Minhaqui da Silva
- **Email:** bruno dot cmsilva at ufpe dot br
- **Data:** Fevereiro de 2025
