# Guía de Uso: Tutor Inteligente RAG (The Drummer OS) 🧠🥁

Este sistema utiliza **Qdrant** como base de datos vectorial y **Ollama (Llama 3)** como cerebro para responder dudas técnicas con contexto local.

## Endpoints de la API

El servicio corre internamente en el cluster en el puerto `8000`.

### 1. Verificar Estado (`GET /status`)
Comprueba que la API puede comunicarse con Ollama y Qdrant.
```bash
curl http://rag-api-service:8000/status
```

### 2. Ingestar Conocimiento (`POST /ingest`)
Envía contenido técnico para que el tutor lo "aprenda".
```bash
curl -X POST http://rag-api-service:8000/ingest \
     -H "Content-Type: application/json" \
     -d '{"text": "SaaS Factory es una plataforma de orquestación de microservicios que utiliza Kubernetes para garantizar alta disponibilidad y PostgreSQL con RLS para aislamiento de datos."}'
```

### 3. Consultar al Tutor (`POST /query`)
Realiza preguntas basadas en el contenido cargado.
```bash
curl -X POST http://rag-api-service:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "¿Cómo garantiza SaaS Factory el aislamiento de datos?"}'
```

## Arquitectura del Sistema
1. **Ingesta**: El texto se convierte en un vector de 4096 dimensiones usando `llama3` en Ollama y se almacena en Qdrant.
2. **Consulta**: La pregunta se vectoriza, se buscan los fragmentos más similares en Qdrant (contexto) y se le pasan a Ollama para generar la respuesta final.

---
> [!TIP]
> Para mejor rendimiento, asegúrate de que Ollama tenga cargado el modelo `llama3`.
