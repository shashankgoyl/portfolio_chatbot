import gc
import logging
import math
import re
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

import config

logger = logging.getLogger(__name__)

RESUME_FILE = os.path.join(os.path.dirname(__file__), "resume.txt")

STOPWORDS = frozenset(
    "what is are the a an do does how can you i me we our tell about for of "
    "in to and or any this that it its be was were have has had will would "
    "could should may might shall just with from by at on".split()
)

SYSTEM_PERSONA = f"""You are a helpful and friendly AI assistant for {config.OWNER_NAME}'s portfolio website.
Answer questions about {config.OWNER_NAME}'s skills, experience, projects, and education.
Be concise, warm, and professional — like a personal assistant.
For hiring or collaboration always mention: {config.OWNER_EMAIL}
For code/projects: {config.OWNER_GITHUB}"""


class RAGSystem:
    """
    Lightweight portfolio RAG — TF-IDF retrieval + Groq LLM.
    Zero heavy dependencies: no FAISS, no embedding model, no torch.
    """

    def __init__(self):
        self.llm: Optional[ChatGroq] = None
        self.chunks: List[Document] = []
        self.rag_mode: bool = False
        self._idf_cache: Dict[str, float] = {}

    def initialize(self) -> None:
        logger.info("🤖 Loading Groq LLM (%s)…", config.GROQ_MODEL)
        self.llm = ChatGroq(
            groq_api_key=config.GROQ_API_KEY,
            model_name=config.GROQ_MODEL,
            temperature=config.GROQ_TEMPERATURE,
            max_tokens=config.GROQ_MAX_TOKENS,
        )
        logger.info("✅ LLM ready")

        logger.info("📄 Loading resume from %s…", RESUME_FILE)
        if not os.path.exists(RESUME_FILE):
            logger.error("❌ resume.txt not found!")
            self.rag_mode = False
            return

        with open(RESUME_FILE, "r", encoding="utf-8") as f:
            raw_text = f.read()

        documents = [Document(page_content=raw_text, metadata={"source": "resume.txt"})]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.RAG_CHUNK_SIZE,
            chunk_overlap=config.RAG_CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.chunks = splitter.split_documents(documents)
        self.chunks = [c for c in self.chunks if len(c.page_content.strip()) > 30]
        logger.info("✅ %d chunks created", len(self.chunks))

        self._build_idf_index()
        self.rag_mode = True
        logger.info("✅ TF-IDF index ready — RAG system online")
        gc.collect()

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-z]{2,}", text.lower())
        return [t for t in tokens if t not in STOPWORDS]

    def _build_idf_index(self) -> None:
        N = len(self.chunks)
        if N == 0:
            return
        df: Counter = Counter()
        for chunk in self.chunks:
            df.update(set(self._tokenize(chunk.page_content)))
        self._idf_cache = {
            term: math.log((N + 1) / (count + 1)) + 1
            for term, count in df.items()
        }
        logger.info("📊 IDF index — %d unique terms", len(self._idf_cache))

    def _tfidf_score(self, query_tokens: List[str], chunk: Document) -> float:
        chunk_tokens = self._tokenize(chunk.page_content)
        if not chunk_tokens:
            return 0.0
        tf_map = Counter(chunk_tokens)
        total = len(chunk_tokens)
        return sum(
            (tf_map.get(t, 0) / total) * self._idf_cache.get(t, 1.0)
            for t in query_tokens
        )

    def _retrieve(self, question: str) -> List[Document]:
        query_tokens = self._tokenize(question)
        if not query_tokens:
            return self.chunks[: config.RAG_TOP_K]
        scored: List[Tuple[float, Document]] = [
            (self._tfidf_score(query_tokens, c), c) for c in self.chunks
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[: config.RAG_TOP_K]]

    def _format_history(self, history: List[Dict]) -> str:
        if not history:
            return ""
        lines = []
        for turn in history[-config.MAX_CONVERSATION_HISTORY:]:
            lines.append(f"User: {turn['question']}")
            lines.append(f"Assistant: {turn['answer']}")
        return "\n".join(lines)

    def _get_rag_response(self, question: str, history: List[Dict]) -> Dict:
        docs = self._retrieve(question)
        context = "\n\n".join(f"[Section]\n{doc.page_content}" for doc in docs)
        history_block = self._format_history(history)
        history_section = f"\nConversation so far:\n{history_block}\n" if history_block else ""

        prompt = f"""{SYSTEM_PERSONA}
{history_section}
---
Relevant info from {config.OWNER_NAME}'s resume:
{context}
---
Visitor question: {question}

Rules:
- Answer ONLY from the context. Do not invent details.
- If not in context, say you don't have that detail and suggest {config.OWNER_EMAIL}.
- Concise and friendly. Bullet points for lists. Max 2 emojis.
- Never repeat the question verbatim.

Answer:"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {"answer": response.content.strip(), "sources": ["resume.txt"], "mode": "rag"}

    def _get_llm_only_response(self, question: str, history: List[Dict]) -> Dict:
        history_block = self._format_history(history)
        history_section = f"\nConversation so far:\n{history_block}" if history_block else ""
        system_msg = f"""{SYSTEM_PERSONA}
{history_section}
No resume data loaded right now. For specifics, direct visitors to {config.OWNER_EMAIL}.
Keep it to 3-4 sentences max."""
        response = self.llm.invoke(
            [SystemMessage(content=system_msg), HumanMessage(content=question)]
        )
        return {"answer": response.content.strip(), "sources": [], "mode": "llm_only"}

    def query(self, question: str, history: Optional[List[Dict]] = None) -> Dict:
        if not self.llm:
            raise RuntimeError("RAGSystem.initialize() has not been called yet.")
        history = history or []
        question = question.strip()
        try:
            if self.rag_mode and self.chunks:
                return self._get_rag_response(question, history)
            return self._get_llm_only_response(question, history)
        except Exception as primary_exc:
            logger.error("Primary query failed: %s", primary_exc)
            try:
                return self._get_llm_only_response(question, history)
            except Exception as fallback_exc:
                logger.error("Fallback failed: %s", fallback_exc)
                return {
                    "answer": f"I'm having trouble right now. Please reach out at {config.OWNER_EMAIL} 🙏",
                    "sources": [],
                    "mode": "error",
                }
