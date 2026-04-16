import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.5"))
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "1024"))

# ── Server ────────────────────────────────────────────────
SERVER_PORT = int(os.getenv("PORT", "8000"))
CORS_ORIGINS = ["*"]  # Restrict in production if needed

# ── RAG ───────────────────────────────────────────────────
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "400"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))

# ── Conversation ──────────────────────────────────────────
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "6"))

# ── Owner Info (personalise this!) ────────────────────────
OWNER_NAME = "Shashank Goyal"
OWNER_EMAIL = "shashankgoyal902@gmail.com"
OWNER_GITHUB = "github.com/shashankgoyl"
OWNER_LINKEDIN = "linkedin.com/in/shashank-goyal-006593203"
OWNER_PORTFOLIO = "shashankgoyal902.netlify.app"
