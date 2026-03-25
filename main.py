import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
from anthropic import Anthropic
import uuid
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Intelligent Search Engine - RAG Service")

# Configuración CORS
origins = [
    "https://agia360.cloud",
    "http://agia360.cloud",
]

# Permitir subdominios dinamicamente
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://([a-zA-Z0-9-]+\.)*agia360\.cloud",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de Entorno
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    print("WARNING: ANTHROPIC_API_KEY is not set.")

# Inicializar Anthropic
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Inicializar ChromaDB (Local en contenedor)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Usaremos el modelo por defecto de Chroma (all-MiniLM-L6-v2) vía sentence-transformers
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

COLLECTION_NAME = "knowledge_base"
try:
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef
    )
except Exception as e:
    print(f"Error inicializando ChromaDB: {e}")
    collection = None


class AskRequest(BaseModel):
    question: str


@app.get("/health")
async def health_check():
    status = "ok"
    details = {"chromadb": "connected", "anthropic": "configured" if ANTHROPIC_API_KEY else "missing_key"}
    
    # Verificar ChromaDB superficialmente
    try:
        if collection is None:
            raise Exception("ChromaDB collection is None.")
        collection.count()
    except Exception as e:
        status = "warning"
        details["chromadb"] = f"disconnected: {str(e)}"

    return {"status": status, "details": details}


@app.post("/upload")
async def upload_knowledge(text: str = Body(..., embed=True)):
    """ Endpoint para vectorizar y guardar texto en ChromaDB """
    if collection is None:
        raise HTTPException(status_code=500, detail="ChromaDB no está inicializado correctamente.")
    
    try:
        # Generar un ID único
        doc_id = str(uuid.uuid4())
        
        # Añadir documento a la colección (Chroma automatiza la vectorización con la embedding function)
        collection.add(
            documents=[text],
            metadatas=[{"source": "upload_endpoint"}],
            ids=[doc_id]
        )
        return {"success": True, "id": doc_id, "message": "Texto ingerido exitosamente."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al ingerir texto: {str(e)}")


@app.post("/ask")
async def ask_question(request: AskRequest):
    """ Cerebro Híbrido: Recuperar contexto de Chroma y generar respuesta con Claude """
    if collection is None:
        raise HTTPException(status_code=500, detail="ChromaDB no está inicializado.")
    
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY no configurada.")

    try:
        # 1. Recuperar contexto (RAG)
        results = collection.query(
            query_texts=[request.question],
            n_results=3  # Traer los 3 fragmentos más relevantes
        )
        
        # Combinar el contexto recuperado
        retrieved_docs = results["documents"][0] if results and results["documents"] else []
        context = "\n\n".join(retrieved_docs)
        
        if not context.strip():
            context = "No se encontró información relevante en la base de datos de conocimiento."

        # 2. Generar respuesta con Claude (Cerebro Híbrido)
        system_prompt = (
            "Eres un Cerebro Híbrido Avanzado, un buscador inteligente. "
            "Debes responder a la pregunta del usuario utilizando ESTRICTAMENTE "
            "el siguiente contexto recuperado de la base de conocimientos.\n\n"
            f"--- CONTEXTO ---\n{context}\n--- FIN DEL CONTEXTO ---\n\n"
            "Si la respuesta no se encuentra en el contexto, indícalo claramente "
            "y no inventes información. Razona paso a paso si la pregunta es compleja. "
            "Responde siempre en español."
        )

        message = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            temperature=0.2, # Baja temperatura para respuestas más precisas y ceñidas al contexto
            system=system_prompt,
            messages=[
                {"role": "user", "content": request.question}
            ]
        )
        
        # Claude Python SDK v0.4+ devuelve un array de bloques de contenido
        response_text = message.content[0].text
        
        return {
            "answer": response_text,
            "context_used": retrieved_docs
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el Cerebro Híbrido: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
