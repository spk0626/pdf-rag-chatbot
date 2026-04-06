import json
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from core.pdf_loader import load_and_chunk_pdf_bytes

try:
    import redis as redis_lib
except ImportError:  # pragma: no cover - optional dependency
    redis_lib = None

load_dotenv()


@dataclass
class SessionState:
    collection_name: str
    chat_history: list[tuple[str, str]]
    last_accessed: float


class HealthInfo(TypedDict):
    status: str
    active_sessions: int
    embedder: str


def _clone_session_state(session_state: SessionState) -> SessionState:
    return SessionState(
        collection_name=session_state.collection_name,
        chat_history=list(session_state.chat_history),
        last_accessed=session_state.last_accessed,
    )


class SessionStore(ABC):
    @abstractmethod
    def save(self, session_id: str, session_state: SessionState) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, session_id: str) -> SessionState | None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, session_id: str) -> SessionState | None:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def cleanup_expired(self) -> list[tuple[str, SessionState]]:
        raise NotImplementedError


class InMemorySessionStore(SessionStore):
    def __init__(self, ttl_seconds: int) -> None:
        self._ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._sessions: dict[str, SessionState] = {}

    def save(self, session_id: str, session_state: SessionState) -> None:
        with self._lock:
            self._cleanup_locked()
            self._sessions[session_id] = _clone_session_state(session_state)

    def get(self, session_id: str) -> SessionState | None:
        with self._lock:
            self._cleanup_locked()
            session = self._sessions.get(session_id)
            return _clone_session_state(session) if session is not None else None

    def delete(self, session_id: str) -> SessionState | None:
        with self._lock:
            return self._sessions.pop(session_id, None)

    def count(self) -> int:
        with self._lock:
            self._cleanup_locked()
            return len(self._sessions)

    def cleanup_expired(self) -> list[tuple[str, SessionState]]:
        with self._lock:
            return self._cleanup_locked()

    def _cleanup_locked(self) -> list[tuple[str, SessionState]]:
        now = time.time()
        expired_session_ids = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.last_accessed > self._ttl_seconds
        ]

        expired_sessions: list[tuple[str, SessionState]] = []
        for session_id in expired_session_ids:
            session = self._sessions.pop(session_id, None)
            if session is not None:
                expired_sessions.append((session_id, session))

        return expired_sessions


class RedisSessionStore(SessionStore):
    def __init__(self, redis_url: str, ttl_seconds: int, key_prefix: str = "pdf_rag:session:") -> None:
        if redis_lib is None:
            raise RuntimeError("redis package is required when REDIS_URL is configured")

        self._ttl_seconds = ttl_seconds
        self._key_prefix = key_prefix
        self._client = redis_lib.Redis.from_url(redis_url, decode_responses=True)

    def save(self, session_id: str, session_state: SessionState) -> None:
        payload = {
            "collection_name": session_state.collection_name,
            "chat_history": session_state.chat_history,
            "last_accessed": session_state.last_accessed,
        }
        self._client.set(self._key(session_id), json.dumps(payload), ex=self._ttl_seconds)

    def get(self, session_id: str) -> SessionState | None:
        raw_payload = self._client.get(self._key(session_id))
        if raw_payload is None:
            return None

        if isinstance(raw_payload, bytes):
            payload_text = raw_payload.decode("utf-8")
        else:
            payload_text = str(raw_payload)

        data = json.loads(payload_text)
        return SessionState(
            collection_name=data["collection_name"],
            chat_history=[tuple(item) for item in data.get("chat_history", [])],
            last_accessed=float(data.get("last_accessed", time.time())),
        )

    def delete(self, session_id: str) -> SessionState | None:
        session = self.get(session_id)
        self._client.delete(self._key(session_id))
        return session

    def count(self) -> int:
        return sum(1 for _ in self._client.scan_iter(match=f"{self._key_prefix}*"))

    def cleanup_expired(self) -> list[tuple[str, SessionState]]:
        return []

    def _key(self, session_id: str) -> str:
        return f"{self._key_prefix}{session_id}"


class RAGService:
    """API-focused RAG service with isolated per-session collections."""

    def __init__(
        self,
        ttl_seconds: int = 60 * 60,
        collection_prefix: str = "pdf_chunks_",
        chroma_path: str | None = None,
        redis_url: str | None = None,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.collection_prefix = collection_prefix
        self._lock = threading.RLock()
        self._embedder_name = "all-MiniLM-L6-v2"

        chroma_path_value = chroma_path or os.environ.get("CHROMA_PERSIST_DIR", ".chroma")
        self._chroma_client = self._build_chroma_client(chroma_path_value)
        self._session_store = self._build_session_store(redis_url)

        self._embedder = SentenceTransformer(self._embedder_name)
        self._groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        llm_model = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")
        self._fallback_models = [
            llm_model,
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
        ]

    def create_session_from_pdf_bytes(self, pdf_bytes: bytes) -> tuple[str, int]:
        chunks = load_and_chunk_pdf_bytes(pdf_bytes)
        if not chunks:
            raise ValueError("Could not extract text from this PDF. Is it a scanned image?")

        session_id = str(uuid.uuid4())
        collection_name = f"{self.collection_prefix}{session_id.replace('-', '')}"
        session_state = SessionState(
            collection_name=collection_name,
            chat_history=[],
            last_accessed=time.time(),
        )

        with self._lock:
            self._cleanup_expired_sessions_locked()
            self._index_chunks(collection_name, chunks)
            self._session_store.save(session_id, session_state)

        return session_id, len(chunks)

    def ask(self, session_id: str, question: str, top_k: int = 3) -> tuple[str, list[str]]:
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty")

        with self._lock:
            self._cleanup_expired_sessions_locked()
            session = self._session_store.get(session_id)
            if session is None:
                raise KeyError("Session not found. Please upload a PDF again.")

            collection = self._chroma_client.get_collection(session.collection_name)
            chunks = self._retrieve(clean_question, collection, top_k=top_k)
            answer = self._generate_response(clean_question, chunks, session.chat_history)
            session.chat_history.append((clean_question, answer))
            session.last_accessed = time.time()
            self._session_store.save(session_id, session)

        return answer, chunks

    def reset_session(self, session_id: str) -> None:
        with self._lock:
            session = self._session_store.delete(session_id)
            if session is not None:
                self._delete_collection_if_exists(session.collection_name)

    def health(self) -> HealthInfo:
        with self._lock:
            active_sessions = self._session_store.count()

        return {
            "status": "ok",
            "active_sessions": active_sessions,
            "embedder": self._embedder_name,
        }

    def _build_chroma_client(self, chroma_path: str | None):
        if chroma_path:
            Path(chroma_path).mkdir(parents=True, exist_ok=True)
            return chromadb.PersistentClient(path=chroma_path)

        return chromadb.Client()

    def _build_session_store(self, redis_url: str | None) -> SessionStore:
        redis_url_value = redis_url or os.environ.get("REDIS_URL")
        if redis_url_value:
            return RedisSessionStore(redis_url_value, ttl_seconds=self.ttl_seconds)

        return InMemorySessionStore(ttl_seconds=self.ttl_seconds)

    def _index_chunks(self, collection_name: str, chunks: list[str]) -> None:
        self._delete_collection_if_exists(collection_name)
        collection = self._chroma_client.create_collection(collection_name)

        embeddings = self._embedder.encode(chunks, show_progress_bar=False).tolist()
        ids = [f"chunk_{i}" for i in range(len(chunks))]

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
        )

    def _retrieve(self, query: str, collection, top_k: int = 3) -> list[str]:
        query_embedding = self._embedder.encode([query], show_progress_bar=False).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=top_k)
        docs = results.get("documents", [])
        return docs[0] if docs else []

    def _generate_response(
        self,
        query: str,
        chunks: list[str],
        chat_history: list[tuple[str, str]],
    ) -> str:
        context = "\n\n".join(
            f"[Source {i + 1}]: {chunk}" for i, chunk in enumerate(chunks)
        )

        history_text = ""
        for user_msg, bot_msg in chat_history[-3:]:
            history_text += f"User: {user_msg}\nAssistant: {bot_msg}\n"

        system_prompt = (
            "You are a helpful assistant that answers questions strictly based on "
            "the document context provided.\n"
            "- Always cite which source your answer comes from, e.g. [Source 1].\n"
            "- If the answer cannot be found in the context, say so clearly.\n"
            "- Be concise, accurate, and friendly."
        )

        user_prompt = (
            f"Document Context:\n{context}\n\n"
            f"Previous Conversation:\n{history_text}\n"
            f"User Question: {query}\n\n"
            "Answer based only on the context above:"
        )

        response = None
        last_error: Exception | None = None
        for model_name in dict.fromkeys(self._fallback_models):
            try:
                response = self._groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=1024,
                )
                break
            except Exception as exc:
                last_error = exc

        if response is None:
            raise RuntimeError(
                f"All configured Groq models failed: {self._fallback_models}. Last error: {last_error}"
            )

        content = response.choices[0].message.content
        if content is None:
            return "I could not generate a response from the model output."

        return content

    def _cleanup_expired_sessions_locked(self) -> None:
        expired_sessions = self._session_store.cleanup_expired()
        for _, session in expired_sessions:
            self._delete_collection_if_exists(session.collection_name)

    def _delete_collection_if_exists(self, collection_name: str) -> None:
        try:
            self._chroma_client.delete_collection(collection_name)
        except Exception:
            pass
