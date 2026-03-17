"""
app.py – FastAPI Backend Server
================================
Provides REST API endpoints for the GitLab Handbook chatbot.

Endpoints:
    GET  /health  → Health check
    POST /ask     → Ask a question, get an answer with sources

Run with:
    uvicorn backend.app:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from backend.config import settings
from backend.rag_engine import RAGEngine

# ── Initialize FastAPI App ──
app = FastAPI(
    title="GitLab Handbook AI Chatbot",
    description="RAG-powered chatbot that answers questions from GitLab's handbook.",
    version="1.0.0"
)

# ── CORS Middleware (allows frontend to call this API) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialize RAG Engine (loaded once at startup) ──
rag_engine = None


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG engine when the server starts."""
    global rag_engine
    try:
        rag_engine = RAGEngine()
        print("✅ RAG Engine initialized successfully")
    except Exception as e:
        print(f"⚠️ RAG Engine initialization failed: {e}")
        print("   The /ask endpoint will not work until the vector DB is built.")


# ── Request / Response Models ──
class QuestionRequest(BaseModel):
    """Request body for the /ask endpoint."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The question to ask about GitLab's handbook",
        examples=["What are GitLab's core values?"]
    )


class AnswerResponse(BaseModel):
    """Response body from the /ask endpoint."""
    answer: str = Field(description="The generated answer")
    sources: list[str] = Field(description="Source URLs from the handbook")
    num_chunks_used: int = Field(description="Number of context chunks used")


# ── API Endpoints ──
@app.get("/health")
async def health_check():
    """Health check endpoint – verify the server is running."""
    return {
        "status": "healthy",
        "service": "GitLab Handbook AI Chatbot",
        "vector_db_ready": rag_engine is not None
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about GitLab's handbook.

    The system retrieves relevant handbook sections and uses Gemini
    to generate an accurate answer with source citations.
    """
    if rag_engine is None:
        raise HTTPException(
            status_code=503,
            detail="RAG Engine not initialized. Please build the vector database first."
        )

    try:
        result = rag_engine.ask(request.question)
        return AnswerResponse(
            answer=result["answer"],
            sources=result["sources"],
            num_chunks_used=result["num_chunks_used"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer: {str(e)}"
        )


@app.get("/")
def home():
    return {"message": "API is running"}

# ── Run directly ──
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=True
    )
