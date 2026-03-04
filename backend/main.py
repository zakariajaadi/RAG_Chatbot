"""
LangServe server — expose the RAG chain via HTTP.

Endpoints generated automatically :
  POST /rag/invoke              → complete answer (str)
  POST /rag/stream              → streaming SSE
  POST /rag/batch               → multiple quesitions in parallel
  GET  /rag/playground          → UI interactive LangServe

  POST /rag-memory/invoke       → complete answer with mem 
  POST /rag-memory/stream       → streaming SSE with mem
  GET  /rag-memory/playground   → UI interactive LangServe

  GET  /docs                    → Swagger UI
"""

import logging
import warnings

import uvicorn
from fastapi import FastAPI
from langserve import add_routes

from config import read_config
from rag import RAG

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# ---------------------------------------------------------------------------
# Initialize RAG — self-contained, no need to wire vsm/retriever manually
# ---------------------------------------------------------------------------

config = read_config()
rag    = RAG(config)

stateless_chain = rag.get_chain()
stateful_chain  = rag.get_chain(memory=True)

# ---------------------------------------------------------------------------
# FastAPI + LangServe
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG API",
    description="RAG served via LangServe",
    version="1.0.0",
)

# Stateless: input is a plain str
add_routes(app, stateless_chain, path="/rag", input_type=str)

# Stateful: input is {"question": str}, session_id passed via config
add_routes(app, stateful_chain, path="/rag-memory")


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)