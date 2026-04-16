import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from rag_system import RAGSystem

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")

# ─────────────────────────────────────────────
# Conversation store  {conv_id: [turns]}
# ─────────────────────────────────────────────
conversation_store: Dict[str, List[Dict]] = {}


def get_history(conversation_id: str) -> List[Dict]:
    history = conversation_store.get(conversation_id, [])
    return history[-config.MAX_CONVERSATION_HISTORY:]


def save_turn(conversation_id: str, question: str, answer: str) -> None:
    conversation_store.setdefault(conversation_id, []).append(
        {"question": question, "answer": answer, "timestamp": time.time()}
    )


# ─────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ─────────────────────────────────────────────
async def _init_rag(app: FastAPI) -> None:
    try:
        logger.info("🔄 RAG initialisation starting…")
        rag = RAGSystem()
        await asyncio.to_thread(rag.initialize)
        app.state.rag_system = rag
        app.state.system_status = "ready"
        logger.info("✅ RAG fully initialised — system ready")
    except Exception as exc:
        logger.error("❌ RAG initialisation failed: %s", exc)
        app.state.system_status = "failed"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting %s Portfolio AI…", config.OWNER_NAME)
    app.state.system_status = "initializing"
    app.state.rag_system = None
    asyncio.create_task(_init_rag(app))
    yield
    logger.info("🛑 Shutting down")


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title=f"{config.OWNER_NAME} — Portfolio AI",
    description="RAG-powered portfolio chatbot",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static HTML frontend
import os
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/ui", StaticFiles(directory=static_dir, html=True), name="static")


# ─────────────────────────────────────────────
# Global error handler
# ─────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = Field(None, description="Omit to start a new conversation")


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    sources: Optional[List[str]] = None
    mode: Optional[str] = None
    processing_time_ms: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    ready: bool
    rag_mode: bool
    total_chunks: int
    owner: str


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/", tags=["Meta"])
async def root():
    return {
        "service": f"{config.OWNER_NAME} Portfolio AI",
        "status": "running",
        "ui": "/ui",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health():
    rag: Optional[RAGSystem] = app.state.rag_system
    return HealthResponse(
        status=app.state.system_status,
        ready=app.state.system_status == "ready",
        rag_mode=rag.rag_mode if rag else False,
        total_chunks=len(rag.chunks) if rag else 0,
        owner=config.OWNER_NAME,
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    status = app.state.system_status

    if status == "initializing":
        return ChatResponse(
            answer="⏳ I'm warming up! Please try again in 30–60 seconds.",
            conversation_id="initializing",
            mode="initializing",
        )

    if status == "failed":
        return ChatResponse(
            answer=f"⚠️ Something went wrong on startup. Please email {config.OWNER_EMAIL}.",
            conversation_id="error",
            mode="error",
        )

    rag_system: RAGSystem = app.state.rag_system
    conversation_id = request.conversation_id or str(uuid.uuid4())
    history = get_history(conversation_id)

    t0 = time.perf_counter()
    try:
        result = await asyncio.to_thread(rag_system.query, request.message, history)
    except Exception as exc:
        logger.error("Chat error for conv %s: %s", conversation_id, exc)
        result = {
            "answer": f"⚠️ Something went wrong. Please try again or email {config.OWNER_EMAIL}.",
            "sources": [],
            "mode": "error",
        }

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    answer = result.get("answer", "")
    sources = result.get("sources", []) or []
    mode = result.get("mode", "unknown")

    save_turn(conversation_id, request.message, answer)
    logger.info("💬 conv=%s mode=%s time=%dms", conversation_id[:8], mode, elapsed_ms)

    return ChatResponse(
        answer=answer,
        conversation_id=conversation_id,
        sources=sources or None,
        mode=mode,
        processing_time_ms=elapsed_ms,
    )


@app.delete("/conversation/{conversation_id}", tags=["Chat"])
async def clear_conversation(conversation_id: str):
    if conversation_id in conversation_store:
        del conversation_store[conversation_id]
        return {"message": "Cleared", "conversation_id": conversation_id}
    raise HTTPException(status_code=404, detail="Conversation not found")


@app.get("/stats", tags=["Meta"])
async def stats():
    rag: Optional[RAGSystem] = app.state.rag_system
    return {
        "active_conversations": len(conversation_store),
        "total_turns": sum(len(v) for v in conversation_store.values()),
        "rag_chunks": len(rag.chunks) if rag else 0,
        "rag_mode": rag.rag_mode if rag else False,
    }


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=config.SERVER_PORT,
        log_level="info",
        access_log=True,
    )
