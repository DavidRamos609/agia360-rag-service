FROM python:3.11-slim

WORKDIR /app

# Instalar herramientas de compilación necesarias para chromadb y otras librerías
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Usaremos la variable de entorno PORT, con valor por defecto 8000
EXPOSE 8000

# El host y port se toman del comando en el archivo python o podemos forzarlo aquí
CMD ["python", "main.py"]
