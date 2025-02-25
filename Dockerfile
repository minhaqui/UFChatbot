# Usar a imagem base do Python 3.12 slim
FROM python:3.12-slim

# Definir o diretório de trabalho
WORKDIR /app

# Instalar ferramentas do sistema necessárias para compilação
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar o requirements.txt para o container
COPY requirements.txt .

# Instalar as dependências listadas no requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código da aplicação para o container
COPY . .

# Definir o comando para rodar a aplicação com gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]