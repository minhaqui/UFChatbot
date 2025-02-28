# Estágio de build
FROM python:3.11-slim AS builder
WORKDIR /app

# Instalar dependências do sistema para compilação
RUN apt-get update && apt-get install -y --no-install-recommends \
    libblas-dev \
    liblapack-dev \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar as dependências (sem --prefix)
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copiar os pacotes instalados para um diretório temporário
RUN mkdir /install && cp -r /usr/local/lib/python3.11/site-packages /install/

# Estágio final
FROM python:3.11-slim
WORKDIR /app

# Instalar dependências de runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libblas3 \
    liblapack3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copiar os pacotes do estágio de build para o local padrão do Python
COPY --from=builder /install/site-packages /usr/local/lib/python3.11/site-packages

# **Copiar também os executáveis instalados (por exemplo, gunicorn)**
COPY --from=builder /usr/local/bin /usr/local/bin

# Copiar os arquivos da aplicação
COPY app/ .

# Comando de execução com Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "--workers", "2", "--log-level", "debug", "main:app"]