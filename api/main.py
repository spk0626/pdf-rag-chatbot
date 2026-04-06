import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.rag_service import RAGService

app = FastAPI(title="PDF RAG API", version="1.0.0")
_service: RAGService | None = None

allowed_origins = os.environ.get(
    "CORS_ALLOW_ORIGINS",
    "http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:3000,http://localhost:3000",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins.split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_service() -> RAGService:
    global _service

    if _service is None:
        _service = RAGService()

    return _service


class AskRequest(BaseModel):
    session_id: str = Field(min_length=1)
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


class UploadResponse(BaseModel):
    session_id: str
    chunks_indexed: int
    message: str


class HealthResponse(BaseModel):
    status: str
    active_sessions: int
    embedder: str


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "PDF RAG API is running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    data = get_service().health()
    return HealthResponse(
        status=data["status"],
        active_sessions=data["active_sessions"],
        embedder=data["embedder"],
    )


@app.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    if file.content_type not in {"application/pdf", "application/x-pdf"}:
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        session_id, chunks_count = get_service().create_session_from_pdf_bytes(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to index PDF: {exc}") from exc

    return UploadResponse(
        session_id=session_id,
        chunks_indexed=chunks_count,
        message="PDF indexed successfully",
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        answer, sources = get_service().ask(request.session_id, request.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {exc}") from exc

    return AskResponse(answer=answer, sources=sources)


@app.post("/reset-session")
def reset_session(session_id: str) -> dict[str, str]:
    get_service().reset_session(session_id)
    return {"message": "Session reset"}
