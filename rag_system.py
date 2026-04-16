import gc
import logging
import os
from typing import Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

import config

logger = logging.getLogger(__name__)

RESUME_FILE = os.path.join(os.path.dirname(__file__), "resume.txt")

SYSTEM_PERSONA = f"""You are a helpful and friendly AI assistant representing {config.OWNER_NAME}'s portfolio website.
Your job is to answer questions about {config.OWNER_NAME}'s skills, experience, projects, education, and background.
Be concise, warm, and professional. Speak as if you are {config.OWNER_NAME}'s personal assistant.
For hiring or collaboration, always mention: {config.OWNER_EMAIL}
For code/projects: {config.OWNER_GITHUB}"""


class RAGSystem:
    """
    Portfolio RAG system using FAISS + HuggingFace embeddings + Groq LLM.
    Fully free — no paid embedding API required.
    """

    def __init__(self):
        self.llm: Optional[ChatGroq] = None
        self.vectorstore: Optional[FAISS] = None
        self.rag_mode: bool = False
        self.chunks: List[Document] = []

    # ──────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────

    def initialize(self) -> None:
        """Load resume, build FAISS index, and set up the LLM."""

        # 1. LLM
        logger.info("🤖 Loading Groq LLM (%s)…", config.GROQ_MODEL)
        self.llm = ChatGroq(
            groq_api_key=config.GROQ_API_KEY,
            model_name=config.GROQ_MODEL,
            temperature=config.GROQ_TEMPERATURE,
            max_tokens=config.GROQ_MAX_TOKENS,
        )
        logger.info("✅ LLM ready")

        # 2. Load resume
        logger.info("📄 Loading resume from %s…", RESUME_FILE)
        if not os.path.exists(RESUME_FILE):
            logger.error("❌ resume.txt not found at %s", RESUME_FILE)
            self.rag_mode = False
            return

        with open(RESUME_FILE, "r", encoding="utf-8") as f:
            raw_text = f.read()

        documents = [Document(page_content=raw_text, metadata={"source": "resume.txt"})]

        # 3. Chunk
        logger.info("✂️  Splitting into chunks…")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.RAG_CHUNK_SIZE,
            chunk_overlap=config.RAG_CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.chunks = splitter.split_documents(documents)
        self.chunks = [c for c in self.chunks if len(c.page_content.strip()) > 30]
        logger.info("✅ %d chunks created", len(self.chunks))

        # 4. Embeddings (free, local, small model ~80 MB)
        logger.info("🔢 Loading HuggingFace embeddings (all-MiniLM-L6-v2)…")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # 5. FAISS vector store
        logger.info("🗂️  Building FAISS index…")
        self.vectorstore = FAISS.from_documents(self.chunks, embeddings)
        self.rag_mode = True
        logger.info("✅ FAISS index ready — RAG system online")

        gc.collect()

    # ──────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────

    def _retrieve(self, question: str, top_k: int = None) -> List[Document]:
        k = top_k or config.RAG_TOP_K
        if self.vectorstore is None:
            return []
        docs = self.vectorstore.similarity_search(question, k=k)
        return docs

    # ──────────────────────────────────────────
    # History helper
    # ──────────────────────────────────────────

    def _format_history(self, history: List[Dict]) -> str:
        if not history:
            return ""
        lines = []
        for turn in history[-config.MAX_CONVERSATION_HISTORY:]:
            lines.append(f"User: {turn['question']}")
            lines.append(f"Assistant: {turn['answer']}")
        return "\n".join(lines)

    # ──────────────────────────────────────────
    # Response generation
    # ──────────────────────────────────────────

    def _get_rag_response(self, question: str, history: List[Dict]) -> Dict:
        relevant_docs = self._retrieve(question)

        context = "\n\n".join(
            f"[Section: {doc.metadata.get('source', 'resume')}]\n{doc.page_content}"
            for doc in relevant_docs
        )

        history_block = self._format_history(history)
        history_section = f"\nConversation so far:\n{history_block}\n" if history_block else ""

        prompt = f"""{SYSTEM_PERSONA}
{history_section}
---
Relevant information from {config.OWNER_NAME}'s resume/portfolio:
{context}
---
Visitor's question: {question}

Instructions:
- Answer ONLY based on the context above. Do NOT invent or guess details.
- If the answer is not in the context, say you don't have that detail and suggest contacting {config.OWNER_EMAIL}.
- Keep your answer concise and friendly.
- Use bullet points for listing multiple items.
- Never repeat the question verbatim.
- Max 1-2 emojis.

Answer:"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        sources = list({doc.metadata.get("source", "resume") for doc in relevant_docs})

        return {
            "answer": response.content.strip(),
            "sources": sources,
            "mode": "rag",
        }

    def _get_llm_only_response(self, question: str, history: List[Dict]) -> Dict:
        history_block = self._format_history(history)
        history_section = f"\nConversation so far:\n{history_block}" if history_block else ""

        system_msg = f"""{SYSTEM_PERSONA}
{history_section}
I don't have detailed resume data loaded right now.
For specific questions about skills, projects, or experience, ask the visitor to check {config.OWNER_PORTFOLIO} or email {config.OWNER_EMAIL}.
Keep responses concise — max 3-4 sentences."""

        response = self.llm.invoke(
            [SystemMessage(content=system_msg), HumanMessage(content=question)]
        )
        return {"answer": response.content.strip(), "sources": [], "mode": "llm_only"}

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def query(self, question: str, history: Optional[List[Dict]] = None) -> Dict:
        if not self.llm:
            raise RuntimeError("RAGSystem.initialize() has not been called yet.")

        history = history or []
        question = question.strip()

        try:
            if self.rag_mode and self.vectorstore:
                return self._get_rag_response(question, history)
            else:
                return self._get_llm_only_response(question, history)

        except Exception as primary_exc:
            logger.error("Primary query failed: %s", primary_exc)
            try:
                return self._get_llm_only_response(question, history)
            except Exception as fallback_exc:
                logger.error("Fallback also failed: %s", fallback_exc)
                return {
                    "answer": (
                        f"I'm having trouble right now. Please reach out directly at {config.OWNER_EMAIL} 🙏"
                    ),
                    "sources": [],
                    "mode": "error",
                }
