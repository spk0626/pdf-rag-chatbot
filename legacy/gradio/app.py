import gradio as gr
import os
from pathlib import Path
import requests
from gradio.themes import Soft

# -- Global state ---------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 120
session_id: str | None = None


# -- Handlers -------------------------------------------------------------------
def upload_pdf(file):
    """Upload a PDF to FastAPI and start a fresh retrieval session."""
    global session_id

    if file is None:
        return "Please select a PDF file first.", []

    # Gradio 4.x passes a file path string; 3.x passes an object with .name
    file_path = file if isinstance(file, str) else file.name
    path = Path(file_path)

    try:
        with path.open("rb") as handle:
            response = requests.post(
                f"{API_BASE_URL}/upload-pdf",
                files={"file": (path.name, handle, "application/pdf")},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

        if response.status_code >= 400:
            error_detail = response.json().get("detail", response.text)
            return f"Upload failed: {error_detail}", []

        payload = response.json()
        session_id = payload["session_id"]
        chunks_indexed = payload.get("chunks_indexed", 0)

        return (
            f"PDF loaded: {chunks_indexed} chunks indexed.\n"
            "You can now ask questions in the chat."
        ), []

    except Exception as e:
        return f"Error loading PDF: {e}", []


def respond(message: str, history: list[dict] | None):
    """Send question to FastAPI and append the model response to chat history."""
    global session_id

    if history is None:
        history = []

    message = message.strip()
    if not message:
        return history, ""

    if session_id is None:
        history.append({"role": "user", "content": message})
        history.append(
            {
                "role": "assistant",
                "content": "Please upload a PDF first using the panel on the left.",
            }
        )
        return history, ""

    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={"session_id": session_id, "question": message},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if response.status_code >= 400:
            error_detail = response.json().get("detail", response.text)
            answer = f"API error: {error_detail}"
        else:
            payload = response.json()
            answer = payload.get("answer", "I could not generate a response.")
    except Exception as e:
        answer = f"Error generating response: {e}"

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    return history, ""


def clear_chat():
    """Reset the server-side session and clear UI conversation history."""
    global session_id

    if session_id is not None:
        try:
            requests.post(
                f"{API_BASE_URL}/reset-session",
                params={"session_id": session_id},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except Exception:
            pass

    session_id = None
    return []


# -- UI -------------------------------------------------------------------------
CHAT_CSS = """
#chatbot .message {
    max-width: 78% !important;
}
"""


with gr.Blocks(title="PDF Q&A Chatbot (Legacy Gradio)") as demo:
    gr.Markdown(
        """
        # PDF Q & A Chatbot (Legacy Gradio Client)
        Upload any PDF and ask questions about it.
        Powered by FastAPI backend.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload Your Document")
            pdf_upload = gr.File(label="Select a PDF", file_types=[".pdf"])
            upload_btn = gr.Button("Load PDF", variant="primary")
            upload_status = gr.Textbox(
                label="Status",
                interactive=False,
                lines=3,
                placeholder="Upload a PDF to get started...",
            )

        with gr.Column(scale=2):
            gr.Markdown("### 2. Ask Questions")
            chatbot = gr.Chatbot(height=420, label="Conversation", elem_id="chatbot")
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask something about your PDF...",
                    label="",
                    scale=5,
                    container=False,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear Chat", variant="secondary")

    upload_btn.click(fn=upload_pdf, inputs=pdf_upload, outputs=[upload_status, chatbot])
    send_btn.click(fn=respond, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])
    msg_input.submit(fn=respond, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])
    clear_btn.click(fn=clear_chat, outputs=chatbot)


if __name__ == "__main__":
    demo.launch(theme=Soft(), css=CHAT_CSS)