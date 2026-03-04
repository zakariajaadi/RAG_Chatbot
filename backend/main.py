"""
LangServe server — expose the RAG chain via HTTP.

Endpoints générés automatiquement :
  POST /rag/invoke       → réponse complète (str)
  POST /rag/stream       → streaming SSE
  POST /rag/batch        → plusieurs questions en parallèle
  GET  /rag/playground   → UI interactive LangServe
  GET  /docs             → Swagger UI
"""

import logging
import warnings

import uvicorn
from fastapi import FastAPI
from langserve import add_routes

from config import read_config
from embedding import get_embedding_model
from vector_store_manager import VectorStoreManager
from retriever import get_retriever
from rag import RAG

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Initialize RAG


config    = read_config()
rag       = RAG(config)
chain     = rag.get_chain()  

# FastAPI + LangServe

app = FastAPI(
    title="RAG API",
    description="RAG served via LangServe",
    version="1.0.0",
)

add_routes(app, chain, path="/rag", input_type=str)


@app.get("/health")
def health():
    return {"status": "ok"}

# Entry-point


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)