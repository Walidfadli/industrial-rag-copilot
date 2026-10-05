from pathlib import Path
import sys

from fastapi import FastAPI
from pydantic import BaseModel

# ------------------------------------------------------------
# PROJECT PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

# ------------------------------------------------------------
# RAG IMPORTS
# ------------------------------------------------------------

from rag import (
    create_qdrant_client,
    create_embedding_model,
    create_reranker,
    create_llm,
    answer_question,
)


# ------------------------------------------------------------
# FASTAPI APP
# ------------------------------------------------------------

app = FastAPI(
    title="Industrial Maintenance Copilot API",
    description=(
        "FastAPI backend for the Industrial Maintenance "
        "RAG Copilot."
    ),
    version="1.0.0",
)


# ------------------------------------------------------------
# REQUEST MODEL
# ------------------------------------------------------------

class QuestionRequest(BaseModel):
    question: str


# ------------------------------------------------------------
# LOAD RAG SYSTEM
# ------------------------------------------------------------

print("Loading RAG system...")

client = create_qdrant_client()
embedding_model = create_embedding_model()
reranker = create_reranker()
llm = create_llm()

print("RAG system loaded.")


# ------------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Industrial Maintenance Copilot API",
        "status": "online",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


# ------------------------------------------------------------
# ASK ENDPOINT
# ------------------------------------------------------------

@app.post("/ask")
def ask_question(request: QuestionRequest):

    if not request.question.strip():
        return {
            "status": "error",
            "message": "Question cannot be empty.",
        }

    result = answer_question(
        query=request.question,
        client=client,
        embedding_model=embedding_model,
        reranker=reranker,
        llm=llm,
    )

    return result
