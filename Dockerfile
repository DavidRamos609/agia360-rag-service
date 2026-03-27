FROM python:3.11-slim

WORKDIR /app

# Instalar herramientas de compilación necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ⭐ DESCARGAR MODELO AQUI (IMPORTANTE)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
