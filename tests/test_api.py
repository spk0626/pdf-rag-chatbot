from fastapi.testclient import TestClient

from api import main


class FakeService:
    def __init__(self) -> None:
        self.reset_calls: list[str] = []

    def health(self) -> dict[str, object]:
        return {
            "status": "ok",
            "active_sessions": 2,
            "embedder": "all-MiniLM-L6-v2",
        }

    def create_session_from_pdf_bytes(self, pdf_bytes: bytes) -> tuple[str, int]:
        assert pdf_bytes.startswith(b"%PDF")
        return "session-123", 4

    def ask(self, session_id: str, question: str) -> tuple[str, list[str]]:
        assert session_id == "session-123"
        assert question == "What is in the PDF?"
        return "It contains a short PDF sample.", ["chunk_1", "chunk_2"]

    def reset_session(self, session_id: str) -> None:
        self.reset_calls.append(session_id)


def _client_with_fake_service() -> tuple[TestClient, FakeService]:
    fake_service = FakeService()
    main._service = fake_service
    return TestClient(main.app), fake_service


def test_health_endpoint_returns_status() -> None:
    client, _ = _client_with_fake_service()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "active_sessions": 2,
        "embedder": "all-MiniLM-L6-v2",
    }


def test_upload_pdf_endpoint_returns_session() -> None:
    client, _ = _client_with_fake_service()

    response = client.post(
        "/upload-pdf",
        files={"file": ("sample.pdf", b"%PDF-1.4\n%Fake PDF content", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "session_id": "session-123",
        "chunks_indexed": 4,
        "message": "PDF indexed successfully",
    }


def test_ask_endpoint_returns_answer_and_sources() -> None:
    client, _ = _client_with_fake_service()

    response = client.post(
        "/ask",
        json={"session_id": "session-123", "question": "What is in the PDF?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "It contains a short PDF sample.",
        "sources": ["chunk_1", "chunk_2"],
    }


def test_reset_session_endpoint_calls_service() -> None:
    client, fake_service = _client_with_fake_service()

    response = client.post("/reset-session", params={"session_id": "session-123"})

    assert response.status_code == 200
    assert response.json() == {"message": "Session reset"}
    assert fake_service.reset_calls == ["session-123"]
